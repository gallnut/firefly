#include <gtest/gtest.h>

#include "firefly/model/registry.h"

TEST(ModelRegistry, CreatesBuiltinQwenArchitectures)
{
    firefly::model::ModelConfig config;

    EXPECT_NE(firefly::model::ModelRegistry::get().create("Qwen2ForCausalLM", config), nullptr);
    EXPECT_NE(firefly::model::ModelRegistry::get().create("Qwen3ForCausalLM", config), nullptr);
}
