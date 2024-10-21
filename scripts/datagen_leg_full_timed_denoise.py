import os
import time
import datetime
import json
import math
import subprocess
import tempfile
import re
from pathlib import Path
import numpy as np
import statistics
from collections import defaultdict
from scipy.stats import qmc
# import renderer module
import volpy


# helper functions
def sample_unit_sphere(sample):
    z = 1.0 - 2.0 * sample[0]
    r = math.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * sample[1]
    return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)


# Sample is a 3d sample from a normal distribution
def sample_inside_unit_sphere(sample, scale):
    x = sample[0]
    y = sample[1]
    z = sample[2]
    magnitude = math.sqrt(x*x + y*y + z*z)
    x /= magnitude
    y /= magnitude
    z /= magnitude

    s = math.cbrt(scale)
    return volpy.vec3(x*s, y*s, z*s)


if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    CREATE_TEST_SET = True
    RUNS = 5

    SAMPLES = 1
    BOUNCES = 4

    # settings
    OUT_PATH = os.path.join(ROOT_DIR, f'generated/leg-gen-timed/full-DENOISE-{SAMPLES}s-{BOUNCES}b-{RUNS}r')
    VOLUME = os.path.join(ROOT_DIR, 'data/Leg/IMG0001.dcm')
    ENVMAP = os.path.join(ROOT_DIR, 'data/table_mountain_2_puresky_1k.hdr')
    TRANSFER_FUNC = "data/lut_leg/full_gen.txt"
    OUT_FORMAT = ".txt"
    N_VIEWS = 512
    MAX_VARIABILITY = 1.0

    WINDOW_LEFT = 0.0
    WINDOW_WIDTH = 1.0
    CUTOFF = 0.200

    DENSITY_SCALE = 1500
    ENV_STRENGTH = 2
    ALBEDO = volpy.vec3(0.9, 0.9, 0.9)
    PHASE = 0
    FOVY = 70
    SEED = 42
    BACKGROUND = False
    TONEMAPPING = True

    #Cut?
    CLIP_MIN = volpy.vec3(0, 0, 0)

    print(OUT_PATH, VOLUME, ENVMAP)

    # ------------------------------------------
    # Render colmap dataset

    # init
    renderer = volpy.Renderer()
    renderer.init()
    renderer.draw()
    os.makedirs(OUT_PATH, exist_ok=True)

    # setup scene
    renderer.volume = volpy.Volume(VOLUME)
    renderer.scale_and_move_to_unit_cube()
    renderer.commit()

    renderer.seed = SEED
    renderer.bounces = BOUNCES
    renderer.albedo = ALBEDO
    renderer.phase = PHASE
    renderer.density_scale = DENSITY_SCALE
    renderer.environment = volpy.Environment(ENVMAP)
    renderer.environment.strength = ENV_STRENGTH
    renderer.show_environment = BACKGROUND
    renderer.tonemapping = TONEMAPPING
    renderer.vol_clip_min = CLIP_MIN
    renderer.cutoff = CUTOFF

    tf = volpy.TransferFunction(TRANSFER_FUNC)
    tf.window_left = WINDOW_LEFT
    tf.window_width = WINDOW_WIDTH
    renderer.transferfunc = tf

    # random sampler
    samplerOut = qmc.Sobol(d=2, seed=SEED+1)
    rng = np.random.default_rng(seed=SEED)

    # Write settings to folder too
    with open(os.path.join(OUT_PATH, 'settings.json'), 'w', encoding='utf-8') as file:
        data = dict(VOLUME=VOLUME, ENVMAP=ENVMAP, BOUNCES=BOUNCES, SAMPLES=SAMPLES,TRANSFER_FUNC=TRANSFER_FUNC, N_VIEWS=N_VIEWS, WINDOW_LEFT=WINDOW_LEFT, WINDOW_WIDTH=WINDOW_WIDTH, CUTOFF=CUTOFF, DENSITY_SCALE=DENSITY_SCALE, ENV_STRENGTH=ENV_STRENGTH, CLIP_MIN=str(CLIP_MIN))
        json.dump(data, file, ensure_ascii=False, indent=4)

    # Create csv file
    csv_path = os.path.join(OUT_PATH, "times.csv")
    runs_header_ms = ','.join(f"run_{i}_ms" for i in range(RUNS))
    runs_header_denoise_ms = ','.join(f"run_{i}_denoise_ms" for i in range(RUNS))
    with open(csv_path, 'w') as csv:
        csv.write(f'{runs_header_ms},{runs_header_denoise_ms},median_ms,median_denoise_ms\n')

    frame_timing_start = 0

    variability = 0
    startTime = time.time()
    startTimeAbs = startTime


    # Warm up
    renderer.render(1024)
    renderer.draw()

    run_data = defaultdict(list)
    run_data_denoise = defaultdict(list)

    with tempfile.TemporaryDirectory() as tempdir:
        temppath = Path(tempdir)

        for i in range(N_VIEWS):
            # setup camera
            bb_min, bb_max = renderer.volume.AABB("density")
            center = bb_min + (bb_max - bb_min) * 0.5
            radius = (bb_max - center).length()

            # Look at random point inside bounding box
            cam_target = center + (sample_inside_unit_sphere(rng.standard_normal(3), rng.random()) * radius * 0.8 * variability)            # Cam target somewhere withoing 80% radius of volume, increasing variability
            cam_unit_sphere_pos = center + sample_unit_sphere(samplerOut.random()[0, 0:2]) * radius
            cam_pos_moved_out = (cam_unit_sphere_pos + ((cam_unit_sphere_pos - center).normalize() * radius * (1.1 - variability)))         # Cam position start point from 2.1 times radius down to 1.1 times radius around center
            cam_pos = cam_pos_moved_out + (sample_inside_unit_sphere(rng.standard_normal(3), rng.random()) * radius * 0.4 * variability)    # randomize camera position by moving up to 0.4 times radius around

            variability = min(MAX_VARIABILITY, variability + 0.01)

            # Advance the loop here so that poses stay in sync when only creating test set (take every 8th view)
            if CREATE_TEST_SET and i % 8 != 0:
                continue
            
            print(f'rendering  view {i+1}/{N_VIEWS}..')

            renderer.cam_pos = cam_pos
            renderer.cam_dir = (cam_target - renderer.cam_pos).normalize()
            renderer.cam_fov = FOVY

            for run in range(RUNS):
                frame_timing_start = time.time_ns()

                # render view
                renderer.render(SAMPLES)
                renderer.draw()

                frame_timing_end = time.time_ns()

                frame_time_ms = (frame_timing_end - frame_timing_start) / (10 ** 6)
                run_data[i].append(frame_time_ms)

            filename = f"view_{i:06}.png"
            filepath = os.path.join(OUT_PATH, filename)
            renderer.save_with_alpha(filepath)

            tempfile = temppath.joinpath(filename + ".pfm")
            # Try to convert to pfm. Only format supported by plain oidnDenoise. Must be little endian
            conv_cmd = f"magick {filepath} -endian LSB {tempfile}"
            subprocess.run(conv_cmd, shell=True, check=True, stdout = subprocess.DEVNULL)

            # Denoise pfm
            denoised_tempfile = tempfile.parent / (filename + "_denoised1.pfm")
            denoise_cmd1 = f"oidnDenoise --ldr {tempfile} -o {denoised_tempfile}"

            for run in range(RUNS):
                result = subprocess.run(denoise_cmd1, shell=True, check=True, text=True, capture_output=True)

                # Extract mssec from oidn command line output
                # Regex pattern to find the msec value after the "Denoising" line
                pattern = r"Denoising.*\n\s*msec=(\d+\.\d+)"
                match = re.search(pattern, result.stdout)
                denoise_time_ms = float(match.group(1))
                run_data_denoise[i].append(denoise_time_ms)

            output = filepath + "_denoised.png"
            back_conv_cmd = f"magick {denoised_tempfile} -define png:compression-level=1 {output}"
            subprocess.run(back_conv_cmd, shell=True, check=True, stdout = subprocess.DEVNULL)

            tempfile.unlink()
            denoised_tempfile.unlink()

            currentTime = time.time()
            diff = currentTime - startTime
            remaining = str(datetime.timedelta(seconds = ((N_VIEWS-i-1) * diff)))
            print(f"Current frame took: {diff:.2f}s, overall: {remaining.split('.')[0]} left")
            startTime = currentTime

    # write views
    with open(csv_path, 'a') as csv:
        for key in run_data:
            line = ','.join(f"{i}" for i in run_data[key])
            line_denoise = ','.join(f"{i}" for i in run_data_denoise[key])
            median_ms = statistics.median(run_data[key])
            median_denoise_ms = statistics.median(run_data_denoise[key])
            if line:
                csv.write(f"{line},{line_denoise},{median_ms},{median_denoise_ms}\n")


    diff = time.time() - startTimeAbs
    print('--------------------')
    print(f"Done, took {str(datetime.timedelta(seconds = diff)).split('.')[0]}")

    renderer.shutdown()