#include <algorithm>

#include "firefly/execution/engine.h"
#include "firefly/execution/trace.h"
#include "firefly/kernels/attention/attention.h"

namespace firefly::execution
{
Status Engine::process_decode(const std::vector<scheduler::SequencePtr>& requests, const device::Context& context)
{
    if (requests.empty()) return {};
    if (speculative_decoder_ != nullptr && speculative_decoder_->can_decode(requests))
    {
        FIREFLY_NVTX_PUSH("Engine_Speculative_Decode");
        Status status = speculative_decoder_->decode(*this, requests, context);
        FIREFLY_NVTX_POP();
        return status;
    }
    int target_batch_size = supported_batch_sizes_.back();
    for (int supported_batch_size : supported_batch_sizes_)
    {
        if (supported_batch_size >= static_cast<int>(requests.size()))
        {
            target_batch_size = supported_batch_size;
            break;
        }
    }

    int max_context_length = 0;
    for (const auto& request : requests)
    {
        max_context_length = std::max(max_context_length, request->context_len);
    }

    bool prefer_split_decode = max_context_length >= 4096;
    bool requires_dynamic_graph =
        !runtime_requirements_.cuda_graph || prefer_split_decode || (options_.kv_cache_format != KVCacheFormat::Int8 &&
                                kernels::get_attention_backend() == kernels::AttentionBackend::FlashInfer);

    if (requires_dynamic_graph || dec_graphs_.find(target_batch_size) == dec_graphs_.end())
    {
        FIREFLY_NVTX_PUSH("Engine_Decode_Dynamic");
        Status status = process_dynamic_decode(requests, max_context_length, prefer_split_decode, context);
        FIREFLY_NVTX_POP();
        return status;
    }

    bool graph_launched = FIREFLY_TRY(process_static_decode_graph(requests, target_batch_size, context));
    if (!graph_launched)
    {
        FIREFLY_NVTX_PUSH("Engine_Decode_Dynamic");
        Status status = process_dynamic_decode(requests, max_context_length, false, context);
        FIREFLY_NVTX_POP();
        return status;
    }
    return {};
}
}  // namespace firefly::execution
