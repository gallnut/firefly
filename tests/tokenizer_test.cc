#include <gtest/gtest.h>

#include "firefly/model/tokenizer.h"

TEST(Tokenizer, MatchesQwenUnicodePretokenization)
{
    firefly::model::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load("qwen3/tokenizer.json").has_value());

    const auto ids = tokenizer.encode(
        "<|im_start|>user\n用三点解释为什么大模型推理算子需要验证数值精度。<|im_end|>\n"
        "<|im_start|>assistant\n");
    const std::vector<int> expected = {151644, 872,    198,    11622,  112055, 104136, 100678, 26288,
                                       104949, 113272, 69103,  44729,  85106,  48927,  111944, 111387,
                                       1773,   151645, 198,    151644, 77091,  198};

    ASSERT_TRUE(ids.has_value()) << ids.error().describe();
    EXPECT_EQ(ids.value(), expected);
}
