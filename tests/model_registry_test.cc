#include <gtest/gtest.h>

#include "firefly/model/registry.h"

TEST(ModelRegistry, CreatesBuiltinQwenArchitectures)
{
    firefly::model::ModelDescriptor descriptor;

    descriptor.architecture = "Qwen2ForCausalLM";
    EXPECT_NE(firefly::model::ModelRegistry::get().create(descriptor), nullptr);
    descriptor.architecture = "Qwen3ForCausalLM";
    EXPECT_NE(firefly::model::ModelRegistry::get().create(descriptor), nullptr);
}

TEST(ModelRegistry, CreatesQwen35WithHybridRuntimeRequirements)
{
    firefly::model::ModelDescriptor descriptor;
    descriptor.architecture = "Qwen3_5ForConditionalGeneration";
    descriptor.config = {
        .dtype = firefly::DType::BF16,
        .hidden_size = 1024,
        .intermediate_size = 3584,
        .num_hidden_layers = 4,
        .num_attention_heads = 8,
        .num_key_value_heads = 2,
        .head_dim = 256,
        .vocab_size = 248320,
        .max_position_embeddings = 262144,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1e7f,
    };
    descriptor.raw_config = R"({"text_config":{
        "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
        "linear_num_value_heads":16,"linear_value_head_dim":128,
        "rope_parameters":{"partial_rotary_factor":0.25},"tie_word_embeddings":true}})";

    auto model = firefly::model::ModelRegistry::get().create(descriptor);
    ASSERT_NE(model, nullptr);
    const auto requirements = model->runtime_requirements();
    EXPECT_EQ(requirements.kv_cache_layer_count, 1);
    EXPECT_EQ(requirements.kv_cache_head_count, 2);
    EXPECT_EQ(requirements.kv_cache_head_dim, 256);
    EXPECT_EQ(requirements.prefill_chunk_limit, 768);
    EXPECT_TRUE(requirements.sequence_state);
    EXPECT_FALSE(requirements.prefix_cache);
    EXPECT_FALSE(requirements.cuda_graph);
}
