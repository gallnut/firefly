#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "firefly/model/config_loader.h"

namespace firefly::model
{

/** @brief Callable that constructs a model from parsed checkpoint metadata. */
using ModelFactory = std::function<Result<std::unique_ptr<Model>>(const ModelDescriptor&)>;

/** @brief Process-wide registry mapping checkpoint architecture names to model factories. */
class ModelRegistry
{
public:
    /**
     * @brief Returns the singleton registry containing built-in and registered factories.
     * @return Process-lifetime registry instance.
     * @threadsafe Initialization is thread-safe; later registration must be externally serialized.
     */
    static ModelRegistry& get();

    /**
     * @brief Registers or replaces the factory for an architecture name.
     * @param architecture_name Exact key read from a model configuration.
     * @param factory Callable that returns a newly owned model instance.
     */
    void register_factory(const std::string& architecture_name, ModelFactory factory);

    /**
     * @brief Constructs a model using the factory selected by `descriptor.architecture`.
     * @param descriptor Parsed model metadata passed to the selected factory.
     * @return Newly owned model implementation or `ErrorCode::NotFound` when no factory is registered.
     */
    Result<std::unique_ptr<Model>> create(const ModelDescriptor& descriptor) const;

private:
    /** @brief Constructs the singleton and registers built-in model factories. */
    ModelRegistry();
    /** @brief Releases registered factory callables during process shutdown. */
    ~ModelRegistry() = default;

    /** @brief Registry singleton state cannot be copied. */
    ModelRegistry(const ModelRegistry&) = delete;
    /** @brief Registry singleton state cannot be copy-assigned. */
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    std::unordered_map<std::string, ModelFactory> factories_; ///< Factory lookup keyed by architecture name.
};

}  // namespace firefly::model
