#include "firefly/model/config_loader.h"

#include <fstream>

#include <nlohmann/json.hpp>

namespace firefly::model
{

Result<ModelDescriptor> load_model_descriptor(const std::string& config_path)
{
    std::ifstream input(config_path);
    if (!input.is_open())
        return unexpected(Error{ErrorCode::Io, "failed to open model config file: " + config_path});

    nlohmann::json json;
    try
    {
        json = nlohmann::json::parse(input);
    }
    catch (const nlohmann::json::exception& exception)
    {
        return unexpected(Error{ErrorCode::Parse, "failed to parse model config " + config_path + ": " +
                                                      std::string(exception.what())});
    }
    ModelDescriptor descriptor;
    descriptor.raw_config = json.dump();
    const auto& text = json.contains("text_config") ? json.at("text_config") : json;
    auto& config = descriptor.config;
    config.hidden_size = text.value("hidden_size", 0);
    config.intermediate_size = text.value("intermediate_size", 0);
    config.num_hidden_layers = text.value("num_hidden_layers", 0);
    config.num_attention_heads = text.value("num_attention_heads", 0);
    config.num_key_value_heads = text.value("num_key_value_heads", config.num_attention_heads);
    if (config.num_attention_heads <= 0)
        return unexpected(Error{ErrorCode::Parse, "model config num_attention_heads must be positive"});
    config.head_dim = text.value("head_dim", config.hidden_size / config.num_attention_heads);
    config.vocab_size = text.value("vocab_size", 0);
    config.max_position_embeddings = text.value("max_position_embeddings", 32768);
    config.rms_norm_eps = text.value("rms_norm_eps", 1e-6);
    config.rope_theta = text.value("rope_theta", text.value("rope_parameters", nlohmann::json::object()).value(
                                                     "rope_theta", 10000.0f));

    std::string torch_dtype = text.value("dtype", text.value("torch_dtype", "float16"));
    if (torch_dtype == "bfloat16" || torch_dtype == "bf16") config.dtype = DType::BF16;
    else if (torch_dtype == "float32" || torch_dtype == "fp32") config.dtype = DType::F32;
    else config.dtype = DType::F16;

    descriptor.architecture = "Qwen2ForCausalLM";
    if (json.contains("architectures") && json["architectures"].is_array() && !json["architectures"].empty())
        descriptor.architecture = json["architectures"][0].get<std::string>();
    if (config.hidden_size <= 0 || config.num_hidden_layers <= 0 || config.vocab_size <= 0 || config.head_dim <= 0)
        return unexpected(Error{ErrorCode::Parse, "model config contains non-positive required dimensions"});
    return descriptor;
}

}  // namespace firefly::model
