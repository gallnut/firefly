#pragma once

#include <nlohmann/json.hpp>

#include "firefly/model/model.h"
namespace firefly::model
{

struct QwenConfig : public ModelConfig
{
    // Any Qwen-specific configs that are not in ModelConfig would go here
};

}  // namespace firefly::model
