#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "firefly/scheduler/sequence_scheduler.h"

namespace firefly::scheduler
{
namespace
{

std::vector<int> make_tokens(int count)
{
    std::vector<int> tokens;
    tokens.reserve(count);
    for (int i = 0; i < count; ++i)
    {
        tokens.push_back(1000 + i);
    }
    return tokens;
}

std::shared_ptr<Sequence> make_sequence(const std::string& id, const std::vector<int>& tokens)
{
    return std::make_shared<Sequence>(id, tokens, 1);
}

TEST(SchedulerPrefixCowTest, FullHitReplaysOneTokenWithPrivateBlockAndKeepsTreeBlocksStable)
{
    SequenceScheduler scheduler;
    scheduler.init(3, 4, 32);

    auto prompt = make_tokens(32);

    auto seed = make_sequence("seed", prompt);
    scheduler.add_sequence(seed);

    auto seed_batch = scheduler.step();
    ASSERT_EQ(seed_batch.failed_sequences.size(), 0);
    ASSERT_EQ(seed_batch.sequences.size(), 1);
    ASSERT_EQ(seed_batch.sequences[0], seed);
    ASSERT_EQ(seed->block_table.size(), 2);

    std::vector<int> cached_blocks = seed->block_table;
    seed->context_len = static_cast<int>(prompt.size());
    seed->generated_tokens.push_back(2000);
    scheduler.finish_sequence(seed);

    auto hit = make_sequence("hit", prompt);
    scheduler.add_sequence(hit);

    auto hit_batch = scheduler.step();
    ASSERT_EQ(hit_batch.failed_sequences.size(), 0);
    ASSERT_EQ(hit_batch.sequences.size(), 1);
    ASSERT_EQ(hit_batch.sequences[0], hit);

    EXPECT_EQ(hit->context_len, static_cast<int>(prompt.size()) - 1);
    EXPECT_EQ(hit->block_table.size(), cached_blocks.size());
    EXPECT_EQ(hit->block_table[0], cached_blocks[0]);
    EXPECT_EQ(hit->prefix_cow_block_index, 1);
    EXPECT_EQ(hit->prefix_cow_source_block, cached_blocks[1]);
    EXPECT_NE(hit->prefix_cow_private_block, cached_blocks[1]);
    EXPECT_EQ(hit->block_table[1], hit->prefix_cow_private_block);
    ASSERT_EQ(hit->owned_blocks.size(), 1);
    EXPECT_EQ(hit->owned_blocks[0], hit->prefix_cow_private_block);

    hit->context_len += 1;
    hit->generated_tokens.push_back(2001);
    scheduler.finish_sequence(hit);

    auto after_finish = make_sequence("after_finish", prompt);
    scheduler.add_sequence(after_finish);

    EXPECT_EQ(after_finish->context_len, static_cast<int>(prompt.size()));
    ASSERT_EQ(after_finish->block_table.size(), cached_blocks.size());
    EXPECT_EQ(after_finish->block_table, cached_blocks);

    auto after_finish_batch = scheduler.step();
    ASSERT_EQ(after_finish_batch.failed_sequences.size(), 0);
    ASSERT_EQ(after_finish_batch.sequences.size(), 1);
    EXPECT_EQ(after_finish_batch.sequences[0], after_finish);
    EXPECT_EQ(after_finish->context_len, static_cast<int>(prompt.size()) - 1);
    EXPECT_EQ(after_finish->prefix_cow_source_block, cached_blocks[1]);
    EXPECT_NE(after_finish->prefix_cow_private_block, cached_blocks[1]);
}

}  // namespace
}  // namespace firefly::scheduler
