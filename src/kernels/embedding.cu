#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>

#include "firefly/kernels.h"

namespace firefly::kernels
{

// Simplified kernel: scalar only, no templates, no fancy alignment logic.
// Using short instead of half to rule out fp16 ABI issues
__global__ void embedding_kernel_simple(const int* input_ids, const short* table, short* output, int total_tokens,
                                        int hidden_size, int vocab_size)
{
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int token_id = input_ids[token_idx];

    // Sanity check token
    if (token_id < 0 || token_id >= vocab_size)
    {
        // Safe fallback logic - or could use a specific invalid token if desired
        token_id = 0;
    }

    size_t embed_offset = static_cast<size_t>(token_id) * hidden_size;
    size_t out_offset = static_cast<size_t>(token_idx) * hidden_size;

    const short* token_embedding = table + embed_offset;
    short*       token_output = output + out_offset;

    // Scalar copy loop
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
    {
        token_output[i] = token_embedding[i];
    }
}

static thread_local cudaStream_t global_kernel_stream = nullptr;

void set_default_stream(cudaStream_t stream) { global_kernel_stream = stream; }

cudaStream_t get_default_stream() { return global_kernel_stream; }

void embedding_lookup(const Tensor& input_ids, const Tensor& embedding_table, Tensor& output)
{
    // input_ids: [batch_size, seq_len]
    // embedding_table: [vocab_size, hidden_size]
    // output: [batch_size, seq_len, hidden_size]
    int batch_size = input_ids.shape()[0];
    int seq_len = input_ids.shape()[1];
    int hidden_size = embedding_table.shape()[1];

    int total_tokens = input_ids.numel();
    int vocab_size = embedding_table.shape()[0];
    int block_size = 256;

    dim3 grid(total_tokens);
    dim3 block(block_size);

    embedding_kernel_simple<<<grid, block, 0, get_default_stream()>>>(
        (const int*)input_ids.data(), (const short*)embedding_table.data(), (short*)output.data(), total_tokens,
        hidden_size, vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error in embedding_lookup launch: " << cudaGetErrorString(err) << " (Code: " << err << ")"
                  << std::endl;
    }
}

}  // namespace firefly::kernels
// dummy
