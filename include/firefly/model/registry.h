#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "firefly/model/config_loader.h"

namespace firefly::model
{

using ModelFactory = std::function<std::unique_ptr<Model>(const ModelDescriptor&)>;

class ModelRegistry
{
public:
    static ModelRegistry& get();

    void register_factory(const std::string& architecture_name, ModelFactory factory);

    std::unique_ptr<Model> create(const ModelDescriptor& descriptor) const;

private:
    ModelRegistry();
    ~ModelRegistry() = default;

    // Disallow copy/move
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    std::unordered_map<std::string, ModelFactory> factories_;
};

}  // namespace firefly::model
