#include "renderer.h"
#include <fstream>


using namespace cppgl;

// -----------------------------------------------------------
// helper funcs

void blit(const Texture2D& tex) {
    static Shader blit_shader = Shader("blit", "shader/quad.vs", "shader/blit.fs");
    blit_shader->bind();
    blit_shader->uniform("tex", tex, 0);
    Quad::draw();
    blit_shader->unbind();
}

void tonemap(const Texture2D& tex, float exposure, float gamma) {
    static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("tex", tex, 0);
    tonemap_shader->uniform("exposure", exposure);
    tonemap_shader->uniform("gamma", gamma);
    Quad::draw();
    tonemap_shader->unbind();
}

// -----------------------------------------------------------
// OpenGL renderer

void RendererOpenGL::init(bool interactive) {
    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();

    // load default environment map
    if (!environment) {
        glm::vec3 color(1.f);
        environment = std::make_shared<Environment>(Texture2D("background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }
    // compile shaders
    if (!trace_shader)
        trace_shader = Shader("trace", "shader/pathtracer_brick.glsl");
    if (!trace_shader_tf)
        trace_shader_tf = Shader("trace_tf", "shader/pathtracer_brick_tf.glsl");

    // setup color texture
    if (!color) {
        const glm::ivec2 res = Context::resolution();
        color = Texture2D("color", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }

    if (!interactive) {
        auto res = Context::resolution();
        initFBO(res.x, res.y);
        bindFBO();
    }
}

void RendererOpenGL::resize(uint32_t w, uint32_t h) {
    if (color) color->resize(w, h);
}

void RendererOpenGL::commit() {
    density_grids.clear();
    emission_grids.clear();
    majorant_emission = 0.f;
    std::cout << "Preparing brick grids for OpenGL..." << std::endl;
    for (const auto& frame : volume->grids) {
        voldata::Volume::GridPtr density_grid = frame.at("density");
        auto brick_grid = voldata::Volume::to_brick_grid(density_grid);
        density_grids.push_back(brick_grid_to_textures(brick_grid));
        voldata::Volume::GridPtr emission_grid;
        for (const auto& name : { "flame", "flames", "temperature" }) {
            if (frame.find(name) != frame.end()) {
                emission_grid = frame.at(name);
                break;
            }
        }
        if (emission_grid) {
            emission_grids.push_back(brick_grid_to_textures(voldata::Volume::to_brick_grid(emission_grid)));
            majorant_emission = std::max(majorant_emission, emission_grid->minorant_majorant().second);
        }
    }
}

void RendererOpenGL::trace() {
    // select shader
    Shader& shader = transferfunc ? trace_shader_tf : trace_shader;

    // bind
    shader->bind();
    color->bind_image(0, GL_READ_WRITE, GL_RGBA32F);

    // Additional shit
    shader->uniform("cutoff", hounsfieldCutoff);

    // uniforms
    uint32_t tex_unit = 0;
    shader->uniform("bounces", bounces);
    shader->uniform("seed", seed);
    shader->uniform("show_environment", show_environment ? 1 : 0);
    shader->uniform("optimization", 0);
    // camera
    shader->uniform("cam_pos", current_camera()->pos);
    shader->uniform("cam_fov", current_camera()->fov_degree);
    shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));
    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    shader->uniform("vol_minorant", min * density_scale);
    shader->uniform("vol_majorant", maj * density_scale);
    shader->uniform("vol_inv_majorant", 1.f / (maj * density_scale));
    shader->uniform("vol_albedo", albedo);
    shader->uniform("vol_phase_g", phase);
    shader->uniform("vol_density_scale", density_scale);
    shader->uniform("vol_emission_scale", emission_scale);
    shader->uniform("vol_emission_norm", majorant_emission > 0.f ? 1.f / majorant_emission : 1.f);
    // density brick grid data
    const BrickGridGL density = density_grids[volume->grid_frame_counter];
    shader->uniform("vol_density_transform", volume->transform * density.transform);
    shader->uniform("vol_density_inv_transform", glm::inverse(volume->transform * density.transform));
    shader->uniform("vol_density_indirection", density.indirection, tex_unit++);
    shader->uniform("vol_density_range", density.range, tex_unit++);
    shader->uniform("vol_density_atlas", density.atlas, tex_unit++);
    // emission brick grid data
    if (volume->grid_frame_counter < emission_grids.size()) {
        const BrickGridGL emission = emission_grids[volume->grid_frame_counter];
        shader->uniform("vol_emission_transform", volume->transform * emission.transform);
        shader->uniform("vol_emission_inv_transform", glm::inverse(volume->transform * emission.transform));
        shader->uniform("vol_emission_indirection", emission.indirection, tex_unit++);
        shader->uniform("vol_emission_range", emission.range, tex_unit++);
        shader->uniform("vol_emission_atlas", emission.atlas, tex_unit++);
    }
    // transfer function
    if (transferfunc) transferfunc->set_uniforms(shader, 4);
    // environment
    shader->uniform("env_transform", environment->transform);
    shader->uniform("env_inv_transform", glm::inverse(environment->transform));
    shader->uniform("env_strength", environment->strength);
    shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    shader->uniform("env_envmap", environment->envmap, tex_unit++);
    shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // trace
    const glm::ivec2 resolution = Context::resolution();
    shader->uniform("current_sample", ++sample);
    shader->uniform("resolution", resolution);
    shader->dispatch_compute(resolution.x, resolution.y);

    // unbind
    color->unbind_image(0);
    shader->unbind();
}

void RendererOpenGL::draw() {
    if (!color) return;
    if (tonemapping)
        tonemap(color, tonemap_exposure, tonemap_gamma);
    else
        blit(color);
}

void RendererOpenGL::reset() {
    sample = 0;
}

BrickGridGL RendererOpenGL::brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& bricks) {
    // create indirection texture

    Texture3D indirection = Texture3D("brick indirection",
            bricks->indirection.stride.x,
            bricks->indirection.stride.y,
            bricks->indirection.stride.z,
            GL_RGB10_A2UI,
            GL_RGBA_INTEGER,
            GL_UNSIGNED_INT_10_10_10_2,
            bricks->indirection.data.data());
    indirection->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    indirection->unbind();
    // create range texture
    Texture3D range = Texture3D("brick range",
            bricks->range.stride.x,
            bricks->range.stride.y,
            bricks->range.stride.z,
            GL_RG16F,
            GL_RG,
            GL_HALF_FLOAT,
            bricks->range.data.data());
    range->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // create min/max mipmaps
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, bricks->range_mipmaps.size());
    for (uint32_t i = 0; i < bricks->range_mipmaps.size(); ++i) {
        glTexImage3D(GL_TEXTURE_3D,
                i + 1,
                GL_RG16F,
                bricks->range_mipmaps[i].stride.x,
                bricks->range_mipmaps[i].stride.y,
                bricks->range_mipmaps[i].stride.z,
                0,
                GL_RG,
                GL_HALF_FLOAT,
                bricks->range_mipmaps[i].data.data());
    }
    range->unbind();
    // create atlas texture
    Texture3D atlas = Texture3D("brick atlas",
            bricks->atlas.stride.x,
            bricks->atlas.stride.y,
            bricks->atlas.stride.z,
            GL_COMPRESSED_RED,
            GL_RED,
            GL_UNSIGNED_BYTE,
            bricks->atlas.data.data());    
    atlas->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    atlas->unbind();
    // return BrickGridGL
    return BrickGridGL{ indirection, range, atlas, bricks->transform };
}

void RendererOpenGL::scale_and_move_to_unit_cube() {
    // compute max AABB over whole volume (animation)
    glm::vec3 bb_min = glm::vec3(FLT_MAX), bb_max = glm::vec3(FLT_MIN);
    for (const auto frame : volume->grids) {
        const auto grid = frame.at("density");
        bb_min = glm::min(bb_min, glm::vec3(grid->transform * glm::vec4(0, 0, 0, 1)));
        bb_max = glm::max(bb_max, glm::vec3(grid->transform * glm::vec4(glm::vec3(grid->index_extent()), 1)));
    }
    // scale to unit cube and move to origin
    const glm::vec3 extent = bb_max - bb_min;
    const float size = fmaxf(extent.x, fmaxf(extent.y, extent.z));
    if (size != 1.f) {
        volume->transform = glm::translate(glm::scale(glm::mat4(1), glm::vec3(1.f / size)), -bb_min - 0.5f * extent);
        density_scale *= size;
    }
}

glm::vec4 interpolated_value_from_lut(std::vector<glm::vec4> lut, float value) {
    if (value < 0.0) value = 0;
    if (value > 1.0) value = 1.0;
    if (lut.size() == 1) return lut[0];
    
    float scaledIndex = value * (lut.size() - 1);
    size_t indexLow = std::floor(scaledIndex);
    size_t indexHigh = std::ceil(scaledIndex);

    if (indexLow == indexHigh) return lut[indexLow];

    float fraction = scaledIndex - indexLow;

    auto interpolatedValue = ((1 - fraction) * lut[indexLow]) + (fraction * lut[indexHigh]);
    
    return interpolatedValue;
}

// Hacky but whatever
float last_alpha = 0;
std::unique_ptr<std::string> RendererOpenGL::lookup_pointdata(voldata::Volume::DenseGridPtr grid, int idx, float x, float y, float z) {
    // Whats a good value here? Would it make sense to accumulate some density along the ray (poor mans raymarching...)? Too costly?
    float ALPHA_THRESHOLD = 0.2f;
    auto density = grid->lookup(glm::uvec3(x,y,z));
    auto rgba = interpolated_value_from_lut(transferfunc->lut, density);

    if (rgba.w + last_alpha > ALPHA_THRESHOLD) {
        auto worldPos = volume->to_world(glm::vec4(x,y,z,1));
        std::stringstream ss;
        //ss << z << " " << density << " " << idx << " " << worldPos.x << " " << worldPos.y << " " << worldPos.z << " " << rgba.x << " " << rgba.y << " " << rgba.z << " " << rgba.w << " 0 0 0" << std::endl;
        ss << idx << " " << worldPos.x << " " << worldPos.y << " " << worldPos.z << " " << (int) (rgba.x * 255) << " " << (int) (rgba.y * 255) << " " << (int) (rgba.z * 255) << " 0 0 0" << std::endl;
        return std::make_unique<std::string>(ss.str());
    }
    last_alpha = rgba.w;
    return nullptr;
}

void RendererOpenGL::to_pointcloud(float density, std::string path) {
    std::cout << "Creating Pointcloud with density " << density << std::endl;

    auto grid = volume->current_grid_dense();
    auto ext = grid->index_extent();
    int idx = 0;
    std::vector<std::string> lines;


    for (float x = 0; x < ext.x; x += density) {
        // XY plane
        for (float y = 0; y < ext.y; y += density) {
            last_alpha = 0;
            // Sample inwards until we find interesting point
            for (float z = 0; z < ext.z; z += density) {

                auto val = lookup_pointdata(grid, idx, x, y, z);
                if (val) {
                    lines.push_back(*val);
                    idx++;
                    break;
                }
            }

            last_alpha = 0;
            // Sample other direction until we find interesting point
            for (float z = ext.z - 1; z >= 0; z -= density) {

                auto val = lookup_pointdata(grid, idx, x, y, z);
                if (val) {
                    lines.push_back(*val);
                    idx++;
                    break;
                }
            }
        }

        // XZ plane
        for (float z = 0; z < ext.z; z += density) {

            last_alpha = 0;
            // Sample inwards until we find interesting point
            for (float y = 0; y < ext.y; y += density) {
                auto val = lookup_pointdata(grid, idx, x, y, z);
                if (val) {
                    lines.push_back(*val);
                    idx++;
                    break;
                }
            }

            last_alpha = 0;
            // Sample other direction until we find interesting point
            for (float y = ext.y - 1; y >= 0; y -= density) {
                auto val = lookup_pointdata(grid, idx, x, y, z);
                if (val) {
                    lines.push_back(*val);
                    idx++;
                    break;
                }
            }
        }
    }

    // YZ plane
    for (float y = 0; y < ext.y; y += density) {
        for (float z = 0; z < ext.z; z += density) {

            last_alpha = 0;
            // Sample inwards until we find interesting point
            for (float x = 0; x < ext.x; x += density) {
                auto val = lookup_pointdata(grid, idx, x, y, z);
                if (val) {
                    lines.push_back(*val);
                    idx++;
                    break;
                }
            }

            last_alpha = 0;
            // Sample other direction until we find interesting point
            for (float x = ext.x - 1; x >= 0; x -= density) {
                auto val = lookup_pointdata(grid, idx, x, y, z);
                if (val) {
                    lines.push_back(*val);
                    idx++;
                    break;
                }
            }
        }
    }

    std::filesystem::path filepath = path;
    std::ofstream file(filepath);
    if (file.is_open()) {
        for (const auto& line : lines) {
            file << line;
        }
    }
    std::cout << lines.size() << " lines pointcloud saved to " << filepath << std::endl;
}

void RendererOpenGL::initFBO(int width, int height) {
    fboSize = glm::ivec2(width, height);

    // Generate and bind framebuffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Create texture to render to
    glGenTextures(1, &fboTexture);
    glBindTexture(GL_TEXTURE_2D, fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTexture, 0);

    // Create and attach depth buffer
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

    // Check framebuffer status
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Error: Framebuffer is not complete!" << std::endl;
    }

    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RendererOpenGL::bindFBO() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, fboSize.x, fboSize.y);
}

void RendererOpenGL::unbindFBO() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    const glm::ivec2 screenSize = Context::resolution();
    glViewport(0, 0, screenSize.x, screenSize.y);
}