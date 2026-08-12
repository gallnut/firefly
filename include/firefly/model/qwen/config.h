#pragma once

#include <nlohmann/json.hpp>

#include "firefly/model/model.h"
namespace firefly::model::qwen
{

/**
 * @brief Qwen-specific configuration extension point.
 *
 * Current Qwen2 and Qwen3 implementations require only fields inherited from
 * `ModelConfig`; the derived type preserves an architecture-specific API boundary.
 */
struct Config : public ModelConfig
{
    // Any Qwen-specific configs that are not in ModelConfig would go here
};

}  // namespace firefly::model::qwen
