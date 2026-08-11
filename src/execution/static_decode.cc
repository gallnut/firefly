#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>

#include "firefly/core/logging.h"
#include "firefly/execution/engine.h"
#include "firefly/execution/trace.h"

namespace firefly::execution
{
bool Engine::process_static_decode_graph(const std::vector<scheduler::SequencePtr>& requests, int target_batch_size,
                                         const device::Context& context)
{
    cudaStream_t stream = context.stream();
    int          batch_size = requests.size();
    auto         graph_data = dec_graphs_.find(target_batch_size);
    if (graph_data == dec_graphs_.end()) return false;
    auto& gd = graph_data->second;

    std::fill(gd.h_input_ids, gd.h_input_ids + target_batch_size, 0);
    std::fill(gd.h_context_lens, gd.h_context_lens + target_batch_size, 0);
    std::fill(gd.h_block_table, gd.h_block_table + target_batch_size * max_context_blocks_,
              decode_scratch_first_block_);
    for (int i = batch_size; i < target_batch_size; ++i)
    {
        gd.h_block_table[i * max_context_blocks_] = decode_scratch_first_block_ + i;
    }

    for (int i = 0; i < batch_size; ++i)
    {
        auto req = requests[i];
        gd.h_input_ids[i] = req->generated_tokens.back();
        gd.h_context_lens[i] = req->context_len;

        for (size_t b = 0; b < req->block_table.size(); ++b)
        {
            gd.h_block_table[i * max_context_blocks_ + b] = req->block_table[b];
        }
    }

    // Copy to pre-allocated static buffers (using explicit stream)
    FIREFLY_NVTX_PUSH("Engine_Decode_Graph_Copy");
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto d0 = std::chrono::high_resolution_clock::now();
#endif
    cudaMemcpyAsync(gd.input_ids.data(), gd.h_input_ids, target_batch_size * sizeof(int), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(gd.context_lens.data(), gd.h_context_lens, target_batch_size * sizeof(int), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(gd.block_table.data(), gd.h_block_table, target_batch_size * max_context_blocks_ * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto d1 = std::chrono::high_resolution_clock::now();
#endif
    FIREFLY_NVTX_POP();

    // Launch the static graph
    FIREFLY_NVTX_PUSH("Engine_Decode_Graph_Launch");
    auto result = gd.graph.launch(stream);
    if (!result)
    {
        FIREFLY_LOG_ERROR("runtime", "decode graph launch failed error={}", result.error().description());
        FIREFLY_NVTX_POP();
        return false;
    }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto d2 = std::chrono::high_resolution_clock::now();
#endif

    // Extract tokens back to host
    cudaMemcpyAsync(gd.h_next_tokens, gd.next_tokens.data(), target_batch_size * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto d3 = std::chrono::high_resolution_clock::now();
#endif

    // Synchronize before reading host elements
    cudaStreamSynchronize(stream);
    FIREFLY_NVTX_POP();
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto d4 = std::chrono::high_resolution_clock::now();
#endif

    for (int i = 0; i < batch_size; ++i)
    {
        auto req = requests[i];
        req->generated_tokens.push_back(gd.h_next_tokens[i]);
        req->context_len += 1;
    }
#ifdef FIREFLY_ENABLE_RUNTIME_PROFILING
    auto d5 = std::chrono::high_resolution_clock::now();

    static double td_h2d = 0, td_launch = 0, td_d2h = 0, td_sync = 0, td_upd = 0;
    td_h2d += std::chrono::duration<double, std::milli>(d1 - d0).count();
    td_launch += std::chrono::duration<double, std::milli>(d2 - d1).count();
    td_d2h += std::chrono::duration<double, std::milli>(d3 - d2).count();
    td_sync += std::chrono::duration<double, std::milli>(d4 - d3).count();
    td_upd += std::chrono::duration<double, std::milli>(d5 - d4).count();
    static int dec_steps = 0;
    dec_steps++;
    if (dec_steps % 50 == 0)
    {
        FIREFLY_LOG_INFO("performance",
                         "static decode profile steps=50 h2d_ms={:.3f} launch_ms={:.3f} d2h_ms={:.3f} "
                         "sync_ms={:.3f} update_ms={:.3f}",
                         td_h2d, td_launch, td_d2h, td_sync, td_upd);
        td_h2d = td_launch = td_d2h = td_sync = td_upd = 0;
    }
#endif
    return true;
}
}  // namespace firefly::execution
