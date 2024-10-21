import os
import sys
import time
import datetime
import json
import numpy as np
from scipy.stats import qmc
# import colmap import/export script
sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap
# import renderer module
import volpy

# helper functions
def sample_unit_sphere(sample):
    import math
    z = 1.0 - 2.0 * sample[0]
    r = math.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * sample[1]
    return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)

if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    # settings
    OUT_PATH = os.path.join(ROOT_DIR, 'generated/leg-gen/muscle-v1_cut_half')
    VOLUME = os.path.join(ROOT_DIR, 'data/Leg/IMG0001.dcm')
    ENVMAP = os.path.join(ROOT_DIR, 'data/table_mountain_2_puresky_1k.hdr')
    TRANSFER_FUNC = "data/lut_leg/muscle_gen.txt"
    OUT_FORMAT = ".txt"
    N_VIEWS = 512

    WINDOW_LEFT = 0.0
    WINDOW_WIDTH = 1.0
    CUTOFF = 0.200

    DENSITY_SCALE = 1500
    ENV_STRENGTH = 2
    SAMPLES = 4096

    ALBEDO = volpy.vec3(0.9, 0.9, 0.9)
    PHASE = 0
    BOUNCES = 100
    FOVY = 70
    SEED = 42
    BACKGROUND = False
    TONEMAPPING = True

    #Cut?
    CLIP_MIN = volpy.vec3(0, 0, 0.48)

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


    cameras = {}
    images = {}
    points3D = {}

    # HACK: write world-space AABB of volume as point3D (pos + rgb) to dataset
    points3D[0] = colmap.Point3D(id=0, xyz=np.array(renderer.volume.AABB("density")[0]), rgb=np.array(renderer.volume.AABB("density")[1]), error=0, image_ids=np.array([]), point2D_idxs=np.array([]))

    # write camera
    cameras[0] = colmap.Camera(id=0, model="SIMPLE_PINHOLE", width=renderer.resolution().x, height=renderer.resolution().y, params=np.array([renderer.colmap_focal_length(), renderer.resolution().x//2, renderer.resolution().y//2]))

    # random sampler
    samplerOut = qmc.Sobol(d=2, seed=SEED+1)
    samplerIn = qmc.Sobol(d=2, seed=SEED+2)

    # Write settings to folder too
    with open(os.path.join(OUT_PATH, 'settings.json'), 'w', encoding='utf-8') as file:
        data = dict(VOLUME=VOLUME, ENVMAP=ENVMAP, SAMPLES=SAMPLES,TRANSFER_FUNC=TRANSFER_FUNC, N_VIEWS=N_VIEWS, WINDOW_LEFT=WINDOW_LEFT, WINDOW_WIDTH=WINDOW_WIDTH, CUTOFF=CUTOFF, DENSITY_SCALE=DENSITY_SCALE, ENV_STRENGTH=ENV_STRENGTH, CLIP_MIN=str(CLIP_MIN))
        json.dump(data, file, ensure_ascii=False, indent=4)


    startTime = time.time()

    # write views
    for i in range(N_VIEWS):
        print(f'rendering {i+1}/{N_VIEWS}..')
        # setup camera
        bb_min, bb_max = renderer.volume.AABB("density")
        center = bb_min + (bb_max - bb_min) * 0.5
        radius = (bb_max - center).length()
        renderer.cam_pos = center + sample_unit_sphere(samplerOut.random()[0, 0:2]) * radius
        renderer.cam_dir = (center + sample_unit_sphere(samplerIn.random()[0, 0:2]) * radius * 0.1 - renderer.cam_pos).normalize()
        renderer.cam_fov = FOVY
        # render view
        renderer.render(SAMPLES)
        renderer.draw()
        # store view
        filename = f"view_{i:06}.png"
        renderer.save_with_alpha(os.path.join(OUT_PATH, filename))
        images[i] = colmap.Image(id=i, qvec=np.array(renderer.colmap_view_rot())[[3, 0, 1, 2]], tvec=np.array(renderer.colmap_view_trans()), camera_id=0, name=filename, xys=np.array([]), point3D_ids=np.array([]))

        currentTime = time.time()
        diff = currentTime - startTime
        remaining = str(datetime.timedelta(seconds = ((N_VIEWS-i-1) * diff)))
        print(f"Current frame took: {diff:.2f}s, overall: {remaining.split('.')[0]} left")
        startTime = currentTime


    print('--------------------')
    print("#cameras:", len(cameras))
    print("#images:", len(images))
    print("#points3D:", len(points3D))

    colmap.write_model(cameras, images, points3D, path=OUT_PATH, ext=OUT_FORMAT)

    renderer.shutdown()
