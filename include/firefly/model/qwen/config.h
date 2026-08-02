#pragma once

#include <nlohmann/json.hpp>

#include "firefly/model/model.h"
namespace firefly::model::qwen
{

struct Config : public ModelConfig
{
    // Any Qwen-specific configs that are not in ModelConfig would go here
};

}  // namespace firefly::model::qwen
