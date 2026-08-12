#include "firefly/model/registry.h"

namespace firefly::model
{

void register_builtin_models(ModelRegistry& registry);

ModelRegistry::ModelRegistry()
{
    register_builtin_models(*this);
}

ModelRegistry& ModelRegistry::get()
{
    static ModelRegistry instance;
    return instance;
}

void ModelRegistry::register_factory(const std::string& architecture_name, ModelFactory factory)
{
    factories_[architecture_name] = std::move(factory);
}

Result<std::unique_ptr<Model>> ModelRegistry::create(const ModelDescriptor& descriptor) const
{
    auto it = factories_.find(descriptor.architecture);
    if (it != factories_.end())
    {
        return it->second(descriptor);
    }
    return unexpected(Error{ErrorCode::NotFound, "unknown model architecture: " + descriptor.architecture});
}

}  // namespace firefly::model
