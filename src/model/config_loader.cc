#include "firefly/model/config_loader.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace firefly::model
{

ModelDescriptor load_model_descriptor(const std::string& config_path)
{
    std::ifstream input(config_path);
    if (!input.is_open()) throw std::runtime_error("Failed to open config file: " + config_path);

    auto json = nlohmann::json::parse(input);
    ModelDescriptor descriptor;
    auto& config = descriptor.config;
    config.hidden_size = json.value("hidden_size", 0);
    config.intermediate_size = json.value("intermediate_size", 0);
    config.num_hidden_layers = json.value("num_hidden_layers", 0);
    config.num_attention_heads = json.value("num_attention_heads", 0);
    config.num_key_value_heads = json.value("num_key_value_heads", config.num_attention_heads);
    config.head_dim = json.value("head_dim", config.hidden_size / config.num_attention_heads);
    config.vocab_size = json.value("vocab_size", 0);
    config.max_position_embeddings = json.value("max_position_embeddings", 32768);
    config.rms_norm_eps = json.value("rms_norm_eps", 1e-6);
    config.rope_theta = json.value("rope_theta", 10000.0f);

    std::string torch_dtype = json.value("torch_dtype", "float16");
    if (torch_dtype == "bfloat16" || torch_dtype == "bf16") config.dtype = DType::BF16;
    else if (torch_dtype == "float32" || torch_dtype == "fp32") config.dtype = DType::F32;
    else config.dtype = DType::F16;

    descriptor.architecture = "Qwen2ForCausalLM";
    if (json.contains("architectures") && json["architectures"].is_array() && !json["architectures"].empty())
        descriptor.architecture = json["architectures"][0].get<std::string>();
    return descriptor;
}

}  // namespace firefly::model
