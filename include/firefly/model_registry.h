#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "firefly/model/model.h"

namespace firefly::model
{

using ModelFactory = std::function<std::unique_ptr<Model>(const ModelConfig&)>;

class ModelRegistry
{
public:
    static ModelRegistry& get();

    void register_factory(const std::string& architecture_name, ModelFactory factory);

    std::unique_ptr<Model> create(const std::string& architecture_name, const ModelConfig& config) const;

private:
    ModelRegistry() = default;
    ~ModelRegistry() = default;

    // Disallow copy/move
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    std::unordered_map<std::string, ModelFactory> factories_;
};

// Helper macro for static registration
#define REGISTER_MODEL(ArchitectureName, ModelClass)                                                                   \
    namespace                                                                                                          \
    {                                                                                                                  \
    struct ModelClass##_Registrar                                                                                      \
    {                                                                                                                  \
        ModelClass##_Registrar()                                                                                       \
        {                                                                                                              \
            ::firefly::model::ModelRegistry::get().register_factory(ArchitectureName,                                  \
                                                                    [](const firefly::model::ModelConfig& config)      \
                                                                    { return std::make_unique<ModelClass>(config); }); \
        }                                                                                                              \
    };                                                                                                                 \
    static ModelClass##_Registrar static_registrar_##ModelClass;                                                       \
    }

}  // namespace firefly::model
