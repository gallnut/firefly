#include "firefly/model_registry.h"

#include <stdexcept>

namespace firefly::model
{

ModelRegistry& ModelRegistry::get()
{
    static ModelRegistry instance;
    return instance;
}

void ModelRegistry::register_factory(const std::string& architecture_name, ModelFactory factory)
{
    factories_[architecture_name] = std::move(factory);
}

std::unique_ptr<Model> ModelRegistry::create(const std::string& architecture_name, const ModelConfig& config) const
{
    auto it = factories_.find(architecture_name);
    if (it != factories_.end())
    {
        return it->second(config);
    }
    throw std::runtime_error("Unknown model architecture: " + architecture_name);
}

}  // namespace firefly::model
