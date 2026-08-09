#pragma once

#include <string>

#include "firefly/model/model.h"

namespace firefly::model
{

struct ModelDescriptor
{
    ModelConfig config;
    std::string architecture;
    std::string raw_config;
};

ModelDescriptor load_model_descriptor(const std::string& config_path);

}  // namespace firefly::model
