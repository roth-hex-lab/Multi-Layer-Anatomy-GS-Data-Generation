#pragma once

#include <cppgl/texture.h>
#include <glm/glm.hpp>

class Volume {
public:
    Volume(); // default construct as invalid volume
    Volume(const std::string& name, size_t w, size_t h, size_t d, float density); // homogeneous
    Volume(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data); // heterogeneous (linear array of w x h x d uchars)
    Volume(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data); // heterogeneous (linear array of w x h x d ushorts)
    Volume(const std::string& name, size_t w, size_t h, size_t d, const float* data); // heterogeneous (linear array of w x h x d floats)
    // TODO construct from file on disk
    virtual ~Volume();

    explicit inline operator bool() const  { return texture->operator bool(); }
    inline operator GLuint() const { return texture->operator GLuint(); }

    void load(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data);
    void load(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data);
    void load(const std::string& name, size_t w, size_t h, size_t d, const float* data);

    // data
    glm::mat4 model;
    float absorbtion_coefficient;
    float scattering_coefficient;
    // TODO emission?
    float max_density;
    std::shared_ptr<Texture3D> texture;
};

// variadic alias for std::make_shared<>(...)
template <class... Args> std::shared_ptr<Volume> make_volume(Args&&... args) {
    return std::make_shared<Volume>(args...);
}