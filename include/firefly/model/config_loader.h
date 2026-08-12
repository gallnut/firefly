#pragma once

#include <string>

#include "firefly/model/model.h"

namespace firefly::model
{

/** @brief Fully parsed model metadata required to construct a registered model implementation. */
struct ModelDescriptor
{
    ModelConfig config;       ///< Architecture-independent execution dimensions and numeric settings.
    std::string architecture; ///< Factory key taken from the first Hugging Face architecture name.
    std::string raw_config;   ///< Original JSON object serialized for architecture-specific parsing.
};

/**
 * @brief Loads a Hugging Face model configuration into Firefly metadata.
 * @param config_path Path to a JSON configuration file.
 * @return Parsed generic settings, architecture key, and original JSON content.
 * @return Parsed descriptor or a structured I/O or parse error.
 */
Result<ModelDescriptor> load_model_descriptor(const std::string& config_path);

}  // namespace firefly::model
