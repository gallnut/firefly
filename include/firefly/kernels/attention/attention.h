#pragma once

#include "firefly/core/tensor.h"
#include "firefly/device/context.h"

#include <vector>

namespace firefly::kernels
{
/** @brief Selects the implementation used by the unified attention dispatcher. */
enum class AttentionBackend
{
    Auto,       ///< Select an implementation from tensor layout and runtime configuration.
    Paged,      ///< Firefly's custom paged-attention kernels.
    Contiguous, ///< Firefly's custom contiguous attention kernel.
    FlashInfer  ///< FlashInfer planning and execution wrappers.
};

/** @brief Optional pre-quantized query representation for integer paged attention. */
struct QuantizedQuery
{
    const Tensor* values = nullptr; ///< Borrowed signed-int8 query tensor.
    const Tensor* scales = nullptr; ///< Borrowed floating-point scale tensor matching query rows.

    /** @brief Returns true when both quantized values and their scales are present. */
    [[nodiscard]] bool valid() const { return values != nullptr && scales != nullptr; }
};

/** @brief Layout, cache metadata, and backend hints consumed by the attention dispatcher. */
struct AttentionOptions
{
    AttentionBackend backend = AttentionBackend::Auto; ///< Requested implementation.
    const int*       block_table = nullptr; ///< Device mapping from logical pages to cache blocks.
    const int*       context_lengths = nullptr; ///< Device context length per batch row.
    const Tensor*    kv_scales = nullptr; ///< Quantized key/value scale cache, if applicable.
    QuantizedQuery   quantized_query; ///< Optional query quantization produced by fused QK preprocessing.
    int              kv_head_count = 0; ///< Number of key/value heads.
    int              max_context_blocks = 0; ///< Block-table row stride.
    int              max_decode_context_length = 0; ///< Longest context represented by a decode batch.
    int              prefill_context_length = -1; ///< Existing context preceding a prefill chunk.
    bool             prefer_split_decode = false; ///< Prefer split-KV reduction for long decode contexts.
    bool             causal = true; ///< Mask future positions in contiguous attention.
};

/**
 * @brief Dispatches contiguous, paged, quantized, or FlashInfer attention asynchronously.
 * @param query Query tensor, normally `[batch, sequence, query_heads, head_dim]`.
 * @param key Contiguous keys or paged key-cache tensor.
 * @param value Contiguous values or paged value-cache tensor.
 * @param output Preallocated output tensor matching query rows and query-head width.
 * @param options Backend and cache metadata.
 * @param context CUDA execution context.
 */
Status attention(Tensor& query, Tensor& key, Tensor& value, Tensor& output, const AttentionOptions& options,
                 const device::Context& context = {});
/**
 * @brief Prepares backend-specific metadata for a paged single-token decode batch.
 * @param context_lens Device array containing one total context length per batch row.
 * @param block_table Device logical-to-physical page table with `max_context_blocks` entries per row.
 * @param batch_size Number of decode rows.
 * @param max_context_blocks Block-table row stride in physical-page identifiers.
 * @param num_query_heads Number of query heads per token.
 * @param num_kv_heads Number of cached key/value heads per token.
 * @param head_dim Scalar width of each attention head.
 * @param context CUDA stream on which backend planning copies are enqueued.
 */
Status prepare_attention_decode(const int* context_lens, const int* block_table, int batch_size,
                                int max_context_blocks, int num_query_heads, int num_kv_heads, int head_dim,
                                const device::Context& context = {});
/**
 * @brief Prepares backend-specific metadata for a uniform-length paged prefill batch.
 * @param block_table Device logical-to-physical page table.
 * @param batch_size Number of equal-length prompt rows.
 * @param sequence_length Number of new query tokens in each row.
 * @param context_length Total key/value context after appending the current chunk.
 * @param max_context_blocks Block-table row stride.
 * @param num_query_heads Number of query heads per token.
 * @param num_kv_heads Number of key/value heads per cached token.
 * @param head_dim Scalar width of each attention head.
 * @param context CUDA stream used by backend planning.
 */
Status prepare_attention_prefill(const int* block_table, int batch_size, int sequence_length, int context_length,
                                 int max_context_blocks, int num_query_heads, int num_kv_heads, int head_dim,
                                 const device::Context& context = {});
/**
 * @brief Plans a ragged FlashInfer prefill from host-side CSR metadata.
 * @param q_indptr Cumulative query-token offsets for every ragged row.
 * @param kv_indptr Cumulative KV-page offsets for every ragged row.
 * @param last_page_len Valid token count in the final KV page of each row.
 * @param max_context_blocks Block-table row stride.
 * @param num_query_heads Number of query heads per token.
 * @param num_kv_heads Number of cached key/value heads per token.
 * @param head_dim Scalar width of each attention head.
 * @param context CUDA stream used for planning and metadata copies.
 * @return `true` when the selected backend produced a launchable ragged plan.
 */
Result<bool> prepare_attention_prefill_ragged(const std::vector<int>& q_indptr, const std::vector<int>& kv_indptr,
                                              const std::vector<int>& last_page_len, int max_context_blocks,
                                              int num_query_heads, int num_kv_heads, int head_dim,
                                              const device::Context& context = {});
/**
 * @brief Launches the most recently planned ragged prefill.
 * @param query Flattened ragged query tensor.
 * @param key_cache Paged key-cache tensor.
 * @param value_cache Paged value-cache tensor.
 * @param output Preallocated flattened attention output.
 * @param block_table Device page table used by the current plan.
 * @param kv_head_count Number of key/value heads per cached token.
 * @param max_context_blocks Block-table row stride.
 * @param scale Multiplicative query-key score scale.
 * @param context CUDA stream used for asynchronous execution.
 * @return `true` when a compatible planned backend dispatched the operation.
 */
Result<bool> launch_attention_prefill_ragged(Tensor& query, Tensor& key_cache, Tensor& value_cache, Tensor& output,
                                             const int* block_table, int kv_head_count, int max_context_blocks,
                                             float scale, const device::Context& context = {});
/**
 * @brief Reserves current-thread decode scratch for a maximum paged shape.
 * @param batch_size Maximum decode rows.
 * @param num_heads Maximum query-head count.
 * @param max_context_blocks Maximum logical pages per row.
 * @param head_dim Attention head width.
 */
Status reserve_paged_decode_scratch(int batch_size, int num_heads, int max_context_blocks, int head_dim);
/**
 * @brief Returns the attention backend selected from environment configuration.
 * @return Process-wide configured backend, including `Auto` when no override is active.
 */
AttentionBackend get_attention_backend();
/**
 * @brief Returns a stable lowercase name for an attention backend.
 * @param backend Backend enumeration value to describe.
 * @return Null-terminated static string suitable for logs and metrics.
 */
const char*      attention_backend_name(AttentionBackend backend);
}  // namespace firefly::kernels
