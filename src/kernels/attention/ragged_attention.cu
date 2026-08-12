#include "firefly/kernels/attention/ragged_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "firefly/device/error.h"
#include "firefly/core/types.h"

namespace firefly::kernels
{
namespace
{
Status dispatch(const Tensor& tensor, auto&& call)
{
    if (tensor.dtype() == DType::BF16) call.template operator()<__nv_bfloat16>();
    else if (tensor.dtype() == DType::F16) call.template operator()<half>();
    else return unexpected(Error{ErrorCode::InvalidArgument, "ragged attention supports only F16/BF16"});
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "launch ragged attention helper"));
    return {};
}

__global__ void fill_positions_kernel(int* positions, const int* seq_offsets, const int* seq_lengths,
                                      const int* context_lens, int batch, int total_tokens)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total_tokens) return;
    for (int seq = 0; seq < batch; ++seq)
    {
        if (flat >= seq_offsets[seq] && flat < seq_offsets[seq + 1])
        {
            positions[flat] = context_lens[seq] + (flat - seq_offsets[seq]);
            return;
        }
    }
}

template <typename scalar_t>
__global__ void gather_single_token_kernel(const scalar_t* source, scalar_t* destination, const int* seq_offsets,
                                           const int* decode_rows, int decode_count, int elements_per_token)
{
    int row_index = blockIdx.x;
    if (row_index >= decode_count) return;
    int token = seq_offsets[decode_rows[row_index]];
    for (int index = threadIdx.x; index < elements_per_token; index += blockDim.x)
        destination[(int64_t)row_index * elements_per_token + index] =
            source[(int64_t)token * elements_per_token + index];
}

template <typename scalar_t>
__global__ void scatter_single_token_kernel(const scalar_t* source, scalar_t* destination, const int* seq_offsets,
                                            const int* decode_rows, int decode_count, int elements_per_token)
{
    int row_index = blockIdx.x;
    if (row_index >= decode_count) return;
    int token = seq_offsets[decode_rows[row_index]];
    for (int index = threadIdx.x; index < elements_per_token; index += blockDim.x)
        destination[(int64_t)token * elements_per_token + index] =
            source[(int64_t)row_index * elements_per_token + index];
}

template <typename scalar_t>
__global__ void gather_prefill_tokens_kernel(const scalar_t* source, scalar_t* destination,
                                             const int* token_indices, int total_rows, int elements_per_token)
{
    int row_index = blockIdx.x;
    if (row_index >= total_rows) return;
    int token = token_indices[row_index];
    for (int index = threadIdx.x; index < elements_per_token; index += blockDim.x)
        destination[(int64_t)row_index * elements_per_token + index] =
            source[(int64_t)token * elements_per_token + index];
}

template <typename scalar_t>
__global__ void scatter_prefill_tokens_kernel(const scalar_t* source, scalar_t* destination,
                                              const int* token_indices, int total_rows, int elements_per_token)
{
    int row_index = blockIdx.x;
    if (row_index >= total_rows) return;
    int token = token_indices[row_index];
    for (int index = threadIdx.x; index < elements_per_token; index += blockDim.x)
        destination[(int64_t)token * elements_per_token + index] =
            source[(int64_t)row_index * elements_per_token + index];
}

template <typename scalar_t>
__global__ void gather_last_hidden_kernel(const scalar_t* source, scalar_t* destination, int hidden,
                                          const int* last_tokens)
{
    int row = blockIdx.x;
    int token = last_tokens[row];
    for (int index = threadIdx.x; index < hidden; index += blockDim.x)
        destination[(int64_t)row * hidden + index] = source[(int64_t)token * hidden + index];
}
}  // namespace

Status fill_ragged_positions(Tensor& positions, const int* seq_offsets, const int* seq_lengths,
                             const int* context_lens, int batch, int total_tokens,
                             const device::Context& context)
{
    fill_positions_kernel<<<(total_tokens + 255) / 256, 256, 0, context.stream()>>>(
        static_cast<int*>(positions.data()), seq_offsets, seq_lengths, context_lens, batch, total_tokens);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return unexpected(device::cuda_error(error, "fill ragged positions"));
    return {};
}

Status gather_decode_tokens(const Tensor& source, Tensor& destination, const int* seq_offsets,
                            const int* decode_rows, int decode_count, int elements_per_token,
                            const device::Context& context)
{
    return dispatch(source, [&]<typename scalar_t>()
    {
        gather_single_token_kernel<scalar_t><<<decode_count, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(source.data()), static_cast<scalar_t*>(destination.data()), seq_offsets,
            decode_rows, decode_count, elements_per_token);
    });
}

Status scatter_decode_tokens(const Tensor& source, Tensor& destination, const int* seq_offsets,
                             const int* decode_rows, int decode_count, int elements_per_token,
                             const device::Context& context)
{
    return dispatch(source, [&]<typename scalar_t>()
    {
        scatter_single_token_kernel<scalar_t><<<decode_count, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(source.data()), static_cast<scalar_t*>(destination.data()), seq_offsets,
            decode_rows, decode_count, elements_per_token);
    });
}

Status gather_prefill_tokens(const Tensor& source, Tensor& destination, const int* token_indices, int total_rows,
                             int elements_per_token, const device::Context& context)
{
    return dispatch(source, [&]<typename scalar_t>()
    {
        gather_prefill_tokens_kernel<scalar_t><<<total_rows, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(source.data()), static_cast<scalar_t*>(destination.data()), token_indices,
            total_rows, elements_per_token);
    });
}

Status scatter_prefill_tokens(const Tensor& source, Tensor& destination, const int* token_indices, int total_rows,
                              int elements_per_token, const device::Context& context)
{
    return dispatch(source, [&]<typename scalar_t>()
    {
        scatter_prefill_tokens_kernel<scalar_t><<<total_rows, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(source.data()), static_cast<scalar_t*>(destination.data()), token_indices,
            total_rows, elements_per_token);
    });
}

Status gather_last_hidden(const Tensor& source, Tensor& destination, int hidden, const int* last_tokens, int rows,
                          const device::Context& context)
{
    return dispatch(source, [&]<typename scalar_t>()
    {
        gather_last_hidden_kernel<scalar_t><<<rows, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(source.data()), static_cast<scalar_t*>(destination.data()), hidden,
            last_tokens);
    });
}

}  // namespace firefly::kernels
