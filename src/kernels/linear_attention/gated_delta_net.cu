#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "firefly/core/types.h"
#include "firefly/device/graph.h"
#include "firefly/kernels/linear_attention/gated_delta_net.h"

namespace firefly::kernels::linear_attention
{

struct GatedDeltaNetWorkspace::Impl
{
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor log_decay;
    Tensor beta_values;
    Tensor key_beta_exp;
    Tensor key_exp_negative;
    Tensor beta_value;
    Tensor cumulative_decay;
    Tensor matrix;
    Tensor inverse;
    Tensor w;
    Tensor u;
    Tensor chunk_states;
    Tensor working_state;
    Tensor projected;
    Tensor new_value;
    Tensor scaled_value;
    Tensor packed_output;
    Tensor scaled_scores;
    std::unordered_map<uint64_t, device::Graph> chunk_graphs;
};

namespace
{
void check_launch(const char* operation);

template <typename T>
struct Scalar;

template <>
struct Scalar<half>
{
    __device__ static float to_float(half value) { return __half2float(value); }
    __device__ static half from_float(float value) { return __float2half(value); }
};

template <>
struct Scalar<__nv_bfloat16>
{
    __device__ static float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
    __device__ static __nv_bfloat16 from_float(float value) { return __float2bfloat16(value); }
};

template <>
struct Scalar<float>
{
    __device__ static float to_float(float value) { return value; }
    __device__ static float from_float(float value) { return value; }
};

template <typename T>
using CudaScalar = Scalar<T>;

constexpr int warp_size = 32;

template <int ARowStride, int AColStride, int BRowStride, int BColStride, int CStride>
__device__ inline void cute_bf16_mma_tile(const __nv_bfloat16* a_data, const __nv_bfloat16* b_data,
                                          float* c_data, int k_elements, bool load_c)
{
    using namespace cute;
    using MmaAtom = SM80_16x8x16_F32BF16BF16F32_TN;
    auto mma = make_tiled_mma(MmaAtom{});
    auto thr_mma = mma.get_slice(threadIdx.x & 31);
    auto gC = make_tensor(make_smem_ptr(c_data),
                          make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<CStride>{}, Int<1>{})));
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.make_fragment_C(tCgC);
    if (load_c)
        copy(tCgC, tCrC);
    else
        clear(tCrC);

    Copy_Atom<UniversalCopy<__nv_bfloat16>, __nv_bfloat16> s2r_atom_a;
    Copy_Atom<UniversalCopy<__nv_bfloat16>, __nv_bfloat16> s2r_atom_b;
    auto s2r_a = make_tiled_copy_A(s2r_atom_a, mma);
    auto s2r_b = make_tiled_copy_B(s2r_atom_b, mma);
    auto thr_copy_a = s2r_a.get_slice(threadIdx.x & 31);
    auto thr_copy_b = s2r_b.get_slice(threadIdx.x & 31);

    for (int k_offset = 0; k_offset < k_elements; k_offset += 16)
    {
        auto sA = make_tensor(
            make_smem_ptr(const_cast<__nv_bfloat16*>(a_data + k_offset * AColStride)),
            make_layout(make_shape(Int<16>{}, Int<16>{}, Int<1>{}),
                        make_stride(Int<ARowStride>{}, Int<AColStride>{}, Int<0>{})));
        auto sB = make_tensor(
            make_smem_ptr(const_cast<__nv_bfloat16*>(b_data + k_offset * BRowStride)),
            make_layout(make_shape(Int<8>{}, Int<16>{}, Int<1>{}),
                        make_stride(Int<BColStride>{}, Int<BRowStride>{}, Int<0>{})));
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
        auto tXsA = thr_copy_a.partition_S(sA(_, _, Int<0>{}));
        auto tXsB = thr_copy_b.partition_S(sB(_, _, Int<0>{}));
        auto tXrA = thr_copy_a.retile_D(tCrA);
        auto tXrB = thr_copy_b.retile_D(tCrB);
        copy(s2r_atom_a, tXsA, tXrA);
        copy(s2r_atom_b, tXsB, tXrB);
        gemm(mma, tCrA, tCrB, tCrC);
    }
    copy(tCrC, tCgC);
}

void reserve_tensor(Tensor& tensor, int64_t elements, DType dtype, const device::Context& context)
{
    if (tensor.data() && tensor.dtype() == dtype && tensor.numel() >= elements) return;
    tensor = Tensor({elements}, dtype, Device::CUDA, context);
}

template <int HeadDimension, int ChunkSize>
void reserve_chunk_workspace(GatedDeltaNetWorkspace::Impl& workspace, int batch, int sequence_length,
                             int head_count, DType scalar_dtype, const device::Context& context)
{
    const int padded_length = ((sequence_length + ChunkSize - 1) / ChunkSize) * ChunkSize;
    const int chunks_per_head = padded_length / ChunkSize;
    const int64_t head_batches = static_cast<int64_t>(batch) * head_count;
    const int64_t packed_elements = head_batches * padded_length * HeadDimension;
    const int64_t matrix_elements = head_batches * chunks_per_head * ChunkSize * ChunkSize;
    const int64_t state_elements = head_batches * chunks_per_head * HeadDimension * HeadDimension;
    const int64_t working_state_elements = head_batches * HeadDimension * HeadDimension;
    const int64_t token_elements = head_batches * padded_length;

    reserve_tensor(workspace.query, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.key, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.value, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.log_decay, token_elements, DType::F32, context);
    reserve_tensor(workspace.beta_values, token_elements, DType::F32, context);
    reserve_tensor(workspace.key_beta_exp, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.key_exp_negative, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.beta_value, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.cumulative_decay, token_elements, DType::F32, context);
    reserve_tensor(workspace.matrix, matrix_elements, DType::F32, context);
    reserve_tensor(workspace.inverse, matrix_elements, scalar_dtype, context);
    reserve_tensor(workspace.w, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.u, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.chunk_states, state_elements, scalar_dtype, context);
    reserve_tensor(workspace.working_state, working_state_elements, DType::F32, context);
    reserve_tensor(workspace.projected, packed_elements, DType::F32, context);
    reserve_tensor(workspace.new_value, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.scaled_value, packed_elements, scalar_dtype, context);
    reserve_tensor(workspace.packed_output, packed_elements, DType::F32, context);
    reserve_tensor(workspace.scaled_scores, matrix_elements, scalar_dtype, context);
}

struct CublasState
{
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;

    CublasState()
    {
        const cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("linear attention GDN failed to create cuBLAS handle status=" +
                                     std::to_string(static_cast<int>(status)));
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    }

    ~CublasState()
    {
        if (handle) cublasDestroy(handle);
    }
};

CublasState& cublas_state(cudaStream_t stream)
{
    static thread_local CublasState state;
    if (state.stream != stream)
    {
        if (cublasSetStream(state.handle, stream) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("linear attention GDN failed to set cuBLAS stream");
        state.stream = stream;
    }
    return state;
}

template <typename scalar_t>
constexpr cudaDataType_t cublas_data_type()
{
    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) return CUDA_R_16BF;
    return CUDA_R_16F;
}

template <typename scalar_t>
void batched_gemm(const scalar_t* a, const scalar_t* b, void* c, int rows, int columns, int inner,
                  bool transpose_a, bool transpose_b, int64_t stride_a, int64_t stride_b, int64_t stride_c,
                  int batch_count, cudaDataType_t output_type, float alpha, float beta, cudaStream_t stream)
{
    const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int leading_b = transpose_b ? inner : columns;
    const int leading_a = transpose_a ? rows : inner;
    const auto input_type = cublas_data_type<scalar_t>();
    auto& state = cublas_state(stream);
    const cublasStatus_t status = cublasGemmStridedBatchedEx(
        state.handle, operation_b, operation_a, columns, rows, inner, &alpha, b, input_type, leading_b, stride_b,
        a, input_type, leading_a, stride_a, &beta, c, output_type, columns, stride_c, batch_count,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("linear attention GDN batched GEMM failed with status " + std::to_string(status));
}

__device__ __forceinline__ float warp_sum(float value)
{
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

template <int Threads>
__device__ __forceinline__ float block_sum(float value)
{
    constexpr int warp_count = (Threads + warp_size - 1) / warp_size;
    __shared__ float partials[warp_count];
    const int lane = threadIdx.x & (warp_size - 1);
    const int warp = threadIdx.x / warp_size;
    value = warp_sum(value);
    if (lane == 0) partials[warp] = value;
    __syncthreads();
    value = threadIdx.x < warp_count ? partials[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) partials[0] = value;
    __syncthreads();
    return partials[0];
}

template <typename scalar_t>
__global__ void causal_convolution_kernel(const scalar_t* input, const scalar_t* weight, scalar_t* states,
                                           const int* state_slots, const int* context_lengths, scalar_t* output,
                                           int sequence_length, int channel_count, int input_stride, int kernel_size)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    if (channel >= channel_count) return;
    const int slot = state_slots[batch];
    scalar_t* state = states + (static_cast<int64_t>(slot) * channel_count + channel) * kernel_size;
    if (!context_lengths || context_lengths[batch] == 0)
        for (int index = 0; index < kernel_size; ++index) state[index] = CudaScalar<scalar_t>::from_float(0.0f);

    for (int token = 0; token < sequence_length; ++token)
    {
        const scalar_t current = input[(static_cast<int64_t>(batch) * sequence_length + token) * input_stride + channel];
        float convolution = CudaScalar<scalar_t>::to_float(current) *
                            CudaScalar<scalar_t>::to_float(weight[channel * kernel_size + kernel_size - 1]);
        for (int tap = 0; tap + 1 < kernel_size; ++tap)
            convolution += CudaScalar<scalar_t>::to_float(state[tap + 1]) *
                           CudaScalar<scalar_t>::to_float(weight[channel * kernel_size + tap]);
        for (int tap = 0; tap + 1 < kernel_size; ++tap) state[tap] = state[tap + 1];
        state[kernel_size - 1] = current;
        const float activated = convolution / (1.0f + expf(-convolution));
        output[(static_cast<int64_t>(batch) * sequence_length + token) * channel_count + channel] =
            CudaScalar<scalar_t>::from_float(activated);
    }
}

template <typename scalar_t>
__global__ void causal_convolution_prefill_kernel(const scalar_t* input, const scalar_t* weight,
                                                   const scalar_t* initial_states, const int* state_slots,
                                                   const int* context_lengths, scalar_t* output,
                                                   int sequence_length, int channel_count, int kernel_size)
{
    const int token = blockIdx.x;
    const int channel = blockIdx.y * blockDim.x + threadIdx.x;
    const int batch = blockIdx.z;
    if (channel >= channel_count) return;
    const int slot = state_slots[batch];
    const scalar_t* state =
        initial_states + (static_cast<int64_t>(slot) * channel_count + channel) * kernel_size;
    const scalar_t current =
        input[(static_cast<int64_t>(batch) * sequence_length + token) * channel_count + channel];
    float convolution = CudaScalar<scalar_t>::to_float(current) *
                        CudaScalar<scalar_t>::to_float(weight[channel * kernel_size + kernel_size - 1]);
    for (int tap = 0; tap + 1 < kernel_size; ++tap)
    {
        const int source_token = token - (kernel_size - 1 - tap);
        float source = 0.0f;
        if (source_token >= 0)
        {
            source = CudaScalar<scalar_t>::to_float(
                input[(static_cast<int64_t>(batch) * sequence_length + source_token) * channel_count + channel]);
        }
        else if (context_lengths && context_lengths[batch] > 0)
        {
            source = CudaScalar<scalar_t>::to_float(state[source_token + kernel_size]);
        }
        convolution += source * CudaScalar<scalar_t>::to_float(weight[channel * kernel_size + tap]);
    }
    const float activated = convolution / (1.0f + expf(-convolution));
    output[(static_cast<int64_t>(batch) * sequence_length + token) * channel_count + channel] =
        CudaScalar<scalar_t>::from_float(activated);
}

template <typename scalar_t, int TokenTile>
__global__ void causal_convolution_prefill_tiled_kernel(const scalar_t* input, const scalar_t* weight,
                                                         const scalar_t* initial_states, const int* state_slots,
                                                         const int* context_lengths, scalar_t* output,
                                                         int sequence_length, int channel_count, int input_stride,
                                                         int kernel_size)
{
    const int channel = blockIdx.y * blockDim.x + threadIdx.x;
    const int batch = blockIdx.z;
    if (channel >= channel_count) return;
    const int token_start = blockIdx.x * TokenTile;
    const int slot = state_slots[batch];
    const scalar_t* channel_weight = weight + channel * kernel_size;
    const scalar_t* state = initial_states +
        (static_cast<int64_t>(slot) * channel_count + channel) * kernel_size;
    float weights[4];
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) weights[tap] = CudaScalar<scalar_t>::to_float(channel_weight[tap]);
#pragma unroll
    for (int offset = 0; offset < TokenTile; ++offset)
    {
        const int token = token_start + offset;
        if (token >= sequence_length) return;
        const int64_t input_index =
            (static_cast<int64_t>(batch) * sequence_length + token) * input_stride + channel;
        const int64_t output_index =
            (static_cast<int64_t>(batch) * sequence_length + token) * channel_count + channel;
        const scalar_t current = input[input_index];
        float convolution = CudaScalar<scalar_t>::to_float(current) * weights[kernel_size - 1];
#pragma unroll
        for (int tap = 0; tap < 3; ++tap)
        {
            const int source_token = token - (kernel_size - 1 - tap);
            float source = 0.0f;
            if (source_token >= 0)
                source = CudaScalar<scalar_t>::to_float(
                    input[(static_cast<int64_t>(batch) * sequence_length + source_token) * input_stride + channel]);
            else if (context_lengths && context_lengths[batch] > 0)
                source = CudaScalar<scalar_t>::to_float(state[source_token + kernel_size]);
            convolution += source * weights[tap];
        }
        output[output_index] =
            CudaScalar<scalar_t>::from_float(convolution / (1.0f + expf(-convolution)));
    }
}

template <typename scalar_t>
__global__ void causal_convolution_save_state_kernel(const scalar_t* input, scalar_t* states,
                                                      const int* state_slots, const int* context_lengths,
                                                      int sequence_length, int channel_count, int input_stride,
                                                      int kernel_size)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    if (channel >= channel_count) return;
    const int slot = state_slots[batch];
    scalar_t* state = states + (static_cast<int64_t>(slot) * channel_count + channel) * kernel_size;
    for (int tap = 0; tap < kernel_size; ++tap)
    {
        const int token = sequence_length - kernel_size + tap;
        if (token >= 0)
            state[tap] = input[(static_cast<int64_t>(batch) * sequence_length + token) * input_stride + channel];
        else if (!context_lengths || context_lengths[batch] == 0)
            state[tap] = CudaScalar<scalar_t>::from_float(0.0f);
        else
            state[tap] = state[token + kernel_size];
    }
}

template <typename scalar_t, typename state_scalar_t, int HeadDimension, bool CacheState = false>
__global__ void gated_delta_net_kernel(const scalar_t* mixed_qkv, const scalar_t* gate, const scalar_t* decay,
                                        const scalar_t* beta, const float* decay_log, const scalar_t* decay_bias,
                                        const float* norm_weight, state_scalar_t* states, const int* state_slots,
                                        const int* context_lengths, scalar_t* output, int sequence_length,
                                        int head_count, int gate_stride, int scalar_stride, float epsilon)
{
    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int value_dimension = threadIdx.x;
    const int slot = state_slots[batch];
    state_scalar_t* global_state = states +
                          ((static_cast<int64_t>(slot) * head_count + head) * HeadDimension * HeadDimension);
    extern __shared__ float cached_state[];
    float* state = cached_state;
    __shared__ float query_values[HeadDimension];
    __shared__ float key_values[HeadDimension];
    __shared__ float normalized_output[HeadDimension];

    for (int key_dimension = 0; key_dimension < HeadDimension; ++key_dimension)
    {
        const int state_index = key_dimension * HeadDimension + value_dimension;
        state[state_index] = !context_lengths || context_lengths[batch] == 0
                                 ? 0.0f
                                 : CudaScalar<state_scalar_t>::to_float(global_state[state_index]);
    }
    __syncthreads();

    const float query_scale = rsqrtf(static_cast<float>(HeadDimension));
    const int projected_width = head_count * HeadDimension * 3;
    for (int token = 0; token < sequence_length; ++token)
    {
        const scalar_t* token_qkv = mixed_qkv +
            (static_cast<int64_t>(batch) * sequence_length + token) * projected_width;
        const int head_offset = head * HeadDimension;
        float query_value = CudaScalar<scalar_t>::to_float(token_qkv[head_offset + value_dimension]);
        float key_value = CudaScalar<scalar_t>::to_float(
            token_qkv[head_count * HeadDimension + head_offset + value_dimension]);
        const float query_square_sum = block_sum<HeadDimension>(query_value * query_value);
        const float key_square_sum = block_sum<HeadDimension>(key_value * key_value);
        query_values[value_dimension] = query_value * rsqrtf(query_square_sum + 1.0e-6f) * query_scale;
        key_values[value_dimension] = key_value * rsqrtf(key_square_sum + 1.0e-6f);
        __syncthreads();

        const int scalar_index = (batch * sequence_length + token) * scalar_stride + head;
        const float decay_input = CudaScalar<scalar_t>::to_float(decay[scalar_index]) +
                                  CudaScalar<scalar_t>::to_float(decay_bias[head]);
        const float softplus = decay_input > 20.0f ? decay_input : log1pf(expf(decay_input));
        const float state_decay = expf(-expf(decay_log[head]) * softplus);
        const float beta_value = 1.0f / (1.0f + expf(-CudaScalar<scalar_t>::to_float(beta[scalar_index])));
        float memory_value = 0.0f;
        for (int key_dimension = 0; key_dimension < HeadDimension; ++key_dimension)
        {
            const int state_index = key_dimension * HeadDimension + value_dimension;
            const float state_value = state[state_index] * state_decay;
            state[state_index] = state_value;
            memory_value += state_value * key_values[key_dimension];
        }
        const float value = CudaScalar<scalar_t>::to_float(
            token_qkv[2 * head_count * HeadDimension + head_offset + value_dimension]);
        const float delta = (value - memory_value) * beta_value;
        float result = 0.0f;
        for (int key_dimension = 0; key_dimension < HeadDimension; ++key_dimension)
        {
            const int state_index = key_dimension * HeadDimension + value_dimension;
            const float state_value = state[state_index] + key_values[key_dimension] * delta;
            state[state_index] = state_value;
            result += state_value * query_values[key_dimension];
        }
        const float recurrent_output = Scalar<scalar_t>::to_float(Scalar<scalar_t>::from_float(result));
        normalized_output[value_dimension] = recurrent_output;
        const float square_sum = block_sum<HeadDimension>(recurrent_output * recurrent_output);
        const float inverse_rms = rsqrtf(square_sum / static_cast<float>(HeadDimension) + epsilon);
        const float gate_value = CudaScalar<scalar_t>::to_float(
            gate[(static_cast<int64_t>(batch) * sequence_length + token) * gate_stride +
                 head * HeadDimension + value_dimension]);
        const float silu_gate = gate_value / (1.0f + expf(-gate_value));
        const float normalized = Scalar<scalar_t>::to_float(
            Scalar<scalar_t>::from_float(normalized_output[value_dimension] * inverse_rms));
        output[((static_cast<int64_t>(batch) * sequence_length + token) * head_count + head) * HeadDimension +
               value_dimension] = CudaScalar<scalar_t>::from_float(normalized * norm_weight[value_dimension] * silu_gate);
        __syncthreads();
    }

    for (int key_dimension = 0; key_dimension < HeadDimension; ++key_dimension)
    {
        const int state_index = key_dimension * HeadDimension + value_dimension;
        global_state[state_index] = CudaScalar<state_scalar_t>::from_float(state[state_index]);
    }
}

template <typename scalar_t, int HeadDimension, int ValueTile, int WarpsPerBlock>
__global__ void gated_delta_net_prefill_kernel(const scalar_t* mixed_qkv, const scalar_t* decay,
                                                const scalar_t* beta, const float* decay_log,
                                                const scalar_t* decay_bias, float* states, const int* state_slots,
                                                const int* context_lengths, scalar_t* recurrent_output,
                                                int sequence_length, int head_count)
{
    constexpr int key_chunks = HeadDimension / warp_size;
    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);
    const int value_offset = (blockIdx.z * WarpsPerBlock + warp) * ValueTile;
    const int slot = state_slots[batch];
    float* state = states +
                   ((static_cast<int64_t>(slot) * head_count + head) * HeadDimension * HeadDimension);
    float state_values[ValueTile][key_chunks];
    const bool clear_state = !context_lengths || context_lengths[batch] == 0;
#pragma unroll
    for (int column = 0; column < ValueTile; ++column)
    {
 #pragma unroll
        for (int chunk = 0; chunk < key_chunks; ++chunk)
        {
            const int key_dimension = chunk * warp_size + lane;
            state_values[column][chunk] =
                clear_state ? 0.0f : state[key_dimension * HeadDimension + value_offset + column];
        }
    }

    const float query_scale = rsqrtf(static_cast<float>(HeadDimension));
    const int projected_width = head_count * HeadDimension * 3;
    for (int token = 0; token < sequence_length; ++token)
    {
        const scalar_t* token_qkv = mixed_qkv +
            (static_cast<int64_t>(batch) * sequence_length + token) * projected_width;
        const int head_offset = head * HeadDimension;
        float query_values[key_chunks];
        float key_values[key_chunks];
        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
#pragma unroll
        for (int chunk = 0; chunk < key_chunks; ++chunk)
        {
            const int key_dimension = chunk * warp_size + lane;
            query_values[chunk] = CudaScalar<scalar_t>::to_float(token_qkv[head_offset + key_dimension]);
            key_values[chunk] = CudaScalar<scalar_t>::to_float(
                token_qkv[head_count * HeadDimension + head_offset + key_dimension]);
            query_square_sum += query_values[chunk] * query_values[chunk];
            key_square_sum += key_values[chunk] * key_values[chunk];
        }
        query_square_sum = __shfl_sync(0xffffffff, warp_sum(query_square_sum), 0);
        key_square_sum = __shfl_sync(0xffffffff, warp_sum(key_square_sum), 0);
        const float inverse_query_norm = rsqrtf(query_square_sum + 1.0e-6f) * query_scale;
        const float inverse_key_norm = rsqrtf(key_square_sum + 1.0e-6f);
#pragma unroll
        for (int chunk = 0; chunk < key_chunks; ++chunk)
        {
            query_values[chunk] *= inverse_query_norm;
            key_values[chunk] *= inverse_key_norm;
        }

        float state_decay = 0.0f;
        float beta_value = 0.0f;
        if (lane == 0)
        {
            const int scalar_index = (batch * sequence_length + token) * head_count + head;
            const float decay_input = CudaScalar<scalar_t>::to_float(decay[scalar_index]) +
                                      CudaScalar<scalar_t>::to_float(decay_bias[head]);
            const float softplus = decay_input > 20.0f ? decay_input : log1pf(expf(decay_input));
            state_decay = expf(-expf(decay_log[head]) * softplus);
            beta_value = 1.0f / (1.0f + expf(-CudaScalar<scalar_t>::to_float(beta[scalar_index])));
        }
        state_decay = __shfl_sync(0xffffffff, state_decay, 0);
        beta_value = __shfl_sync(0xffffffff, beta_value, 0);
#pragma unroll
        for (int column = 0; column < ValueTile; ++column)
        {
            float memory_value = 0.0f;
#pragma unroll
            for (int chunk = 0; chunk < key_chunks; ++chunk)
            {
                state_values[column][chunk] *= state_decay;
                memory_value += state_values[column][chunk] * key_values[chunk];
            }
            memory_value = __shfl_sync(0xffffffff, warp_sum(memory_value), 0);
            float delta = 0.0f;
            if (lane == 0)
            {
                const float value = CudaScalar<scalar_t>::to_float(
                    token_qkv[2 * head_count * HeadDimension + head_offset + value_offset + column]);
                delta = (value - memory_value) * beta_value;
            }
            delta = __shfl_sync(0xffffffff, delta, 0);
            float result = 0.0f;
#pragma unroll
            for (int chunk = 0; chunk < key_chunks; ++chunk)
            {
                state_values[column][chunk] += key_values[chunk] * delta;
                result += state_values[column][chunk] * query_values[chunk];
            }
            result = warp_sum(result);
            if (lane == 0)
            {
                const int output_index =
                    ((static_cast<int64_t>(batch) * sequence_length + token) * head_count + head) * HeadDimension +
                    value_offset + column;
                recurrent_output[output_index] = CudaScalar<scalar_t>::from_float(result);
            }
        }
    }

#pragma unroll
    for (int column = 0; column < ValueTile; ++column)
    {
#pragma unroll
        for (int chunk = 0; chunk < key_chunks; ++chunk)
        {
            const int key_dimension = chunk * warp_size + lane;
            state[key_dimension * HeadDimension + value_offset + column] = state_values[column][chunk];
        }
    }
}

template <typename scalar_t, int HeadDimension>
__global__ void gated_delta_net_output_kernel(const scalar_t* gate, const float* norm_weight, scalar_t* output,
                                               int row_count, float epsilon)
{
    const int row = blockIdx.x;
    if (row >= row_count) return;
    const int dimension = threadIdx.x;
    const int64_t index = static_cast<int64_t>(row) * HeadDimension + dimension;
    const float recurrent_value = CudaScalar<scalar_t>::to_float(output[index]);
    const float square_sum = block_sum<HeadDimension>(recurrent_value * recurrent_value);
    const float inverse_rms = rsqrtf(square_sum / static_cast<float>(HeadDimension) + epsilon);
    const float gate_value = CudaScalar<scalar_t>::to_float(gate[index]);
    const float silu_gate = gate_value / (1.0f + expf(-gate_value));
    const float normalized = CudaScalar<scalar_t>::to_float(
        CudaScalar<scalar_t>::from_float(recurrent_value * inverse_rms));
    output[index] = CudaScalar<scalar_t>::from_float(normalized * norm_weight[dimension] * silu_gate);
}

template <typename scalar_t, int HeadDimension>
__global__ void gdn_pack_scalar_kernel(const scalar_t* mixed_qkv, const scalar_t* decay, const scalar_t* beta,
                                       const float* decay_log, const scalar_t* decay_bias, scalar_t* query,
                                       scalar_t* key, scalar_t* value, float* log_decay, float* beta_values,
                                       int sequence_length, int padded_length, int head_count, int scalar_stride)
{
    const int row = blockIdx.x;
    const int head = row % head_count;
    const int token_index = row / head_count;
    const int batch = token_index / sequence_length;
    const int token = token_index % sequence_length;
    const int dimension = threadIdx.x;
    const int projected_width = head_count * HeadDimension * 3;
    const scalar_t* token_qkv = mixed_qkv + static_cast<int64_t>(token_index) * projected_width;
    const int head_offset = head * HeadDimension;
    const float query_value = CudaScalar<scalar_t>::to_float(token_qkv[head_offset + dimension]);
    const float key_value = CudaScalar<scalar_t>::to_float(
        token_qkv[head_count * HeadDimension + head_offset + dimension]);
    const float query_square_sum = block_sum<HeadDimension>(query_value * query_value);
    const float key_square_sum = block_sum<HeadDimension>(key_value * key_value);
    const int64_t packed_index =
        ((static_cast<int64_t>(batch) * head_count + head) * padded_length + token) * HeadDimension + dimension;
    query[packed_index] = CudaScalar<scalar_t>::from_float(
        query_value * rsqrtf(query_square_sum + 1.0e-6f) * rsqrtf(static_cast<float>(HeadDimension)));
    key[packed_index] = CudaScalar<scalar_t>::from_float(key_value * rsqrtf(key_square_sum + 1.0e-6f));
    value[packed_index] = token_qkv[2 * head_count * HeadDimension + head_offset + dimension];
    if (dimension == 0)
    {
        const int scalar_index = token_index * scalar_stride + head;
        const float decay_input = CudaScalar<scalar_t>::to_float(decay[scalar_index]) +
                                  CudaScalar<scalar_t>::to_float(decay_bias[head]);
        const float softplus = decay_input > 20.0f ? decay_input : log1pf(expf(decay_input));
        const int64_t packed_scalar =
            (static_cast<int64_t>(batch) * head_count + head) * padded_length + token;
        log_decay[packed_scalar] = -expf(decay_log[head]) * softplus;
        beta_values[packed_scalar] =
            1.0f / (1.0f + expf(-CudaScalar<scalar_t>::to_float(beta[scalar_index])));
    }
}

template <typename scalar_t>
union alignas(16) Vector128
{
    uint4 vector;
    scalar_t elements[16 / sizeof(scalar_t)];
};

template <typename scalar_t, int HeadDimension>
__global__ void gdn_pack_vector_kernel(const scalar_t* mixed_qkv, const scalar_t* decay, const scalar_t* beta,
                                       const float* decay_log, const scalar_t* decay_bias, scalar_t* query,
                                       scalar_t* key, scalar_t* value, float* log_decay, float* beta_values,
                                       int sequence_length, int padded_length, int head_count, int scalar_stride)
{
    constexpr int elements_per_vector = 16 / sizeof(scalar_t);
    constexpr int vectors_per_head = HeadDimension / elements_per_vector;
    static_assert(HeadDimension % elements_per_vector == 0);
    static_assert(vectors_per_head <= warp_size);

    const int row = blockIdx.x;
    const int head = row % head_count;
    const int token_index = row / head_count;
    const int batch = token_index / sequence_length;
    const int token = token_index % sequence_length;
    const int lane = threadIdx.x;
    const int projected_width = head_count * HeadDimension * 3;
    const scalar_t* token_qkv = mixed_qkv + static_cast<int64_t>(token_index) * projected_width;
    const int vector_offset = lane * elements_per_vector;
    const int head_offset = head * HeadDimension + vector_offset;

    Vector128<scalar_t> query_input{};
    Vector128<scalar_t> key_input{};
    Vector128<scalar_t> value_input{};
    float query_square_sum = 0.0f;
    float key_square_sum = 0.0f;
    if (lane < vectors_per_head)
    {
        query_input.vector = *reinterpret_cast<const uint4*>(token_qkv + head_offset);
        key_input.vector =
            *reinterpret_cast<const uint4*>(token_qkv + head_count * HeadDimension + head_offset);
        value_input.vector =
            *reinterpret_cast<const uint4*>(token_qkv + 2 * head_count * HeadDimension + head_offset);
#pragma unroll
        for (int element = 0; element < elements_per_vector; ++element)
        {
            const float query_value = CudaScalar<scalar_t>::to_float(query_input.elements[element]);
            const float key_value = CudaScalar<scalar_t>::to_float(key_input.elements[element]);
            query_square_sum += query_value * query_value;
            key_square_sum += key_value * key_value;
        }
    }
    query_square_sum = __shfl_sync(0xffffffff, warp_sum(query_square_sum), 0);
    key_square_sum = __shfl_sync(0xffffffff, warp_sum(key_square_sum), 0);

    if (lane < vectors_per_head)
    {
        const float query_scale =
            rsqrtf(query_square_sum + 1.0e-6f) * rsqrtf(static_cast<float>(HeadDimension));
        const float key_scale = rsqrtf(key_square_sum + 1.0e-6f);
        Vector128<scalar_t> normalized_query{};
        Vector128<scalar_t> normalized_key{};
#pragma unroll
        for (int element = 0; element < elements_per_vector; ++element)
        {
            normalized_query.elements[element] = CudaScalar<scalar_t>::from_float(
                CudaScalar<scalar_t>::to_float(query_input.elements[element]) * query_scale);
            normalized_key.elements[element] = CudaScalar<scalar_t>::from_float(
                CudaScalar<scalar_t>::to_float(key_input.elements[element]) * key_scale);
        }
        const int64_t packed_index =
            ((static_cast<int64_t>(batch) * head_count + head) * padded_length + token) * HeadDimension +
            vector_offset;
        *reinterpret_cast<uint4*>(query + packed_index) = normalized_query.vector;
        *reinterpret_cast<uint4*>(key + packed_index) = normalized_key.vector;
        *reinterpret_cast<uint4*>(value + packed_index) = value_input.vector;
    }
    if (lane == 0)
    {
        const int scalar_index = token_index * scalar_stride + head;
        const float decay_input = CudaScalar<scalar_t>::to_float(decay[scalar_index]) +
                                  CudaScalar<scalar_t>::to_float(decay_bias[head]);
        const float softplus = decay_input > 20.0f ? decay_input : log1pf(expf(decay_input));
        const int64_t packed_scalar =
            (static_cast<int64_t>(batch) * head_count + head) * padded_length + token;
        log_decay[packed_scalar] = -expf(decay_log[head]) * softplus;
        beta_values[packed_scalar] =
            1.0f / (1.0f + expf(-CudaScalar<scalar_t>::to_float(beta[scalar_index])));
    }
}

template <typename scalar_t, int HeadDimension, int ChunkSize>
__global__ void gdn_chunk_factors_kernel(const scalar_t* key, const scalar_t* value, const float* log_decay,
                                          const float* beta, scalar_t* key_beta_exp, scalar_t* key_exp_negative,
                                          scalar_t* beta_value, float* cumulative_decay, int sequence_length,
                                          int padded_length, int chunks_per_head)
{
    const int matrix = blockIdx.x;
    const int head_batch = matrix / chunks_per_head;
    const int chunk = matrix % chunks_per_head;
    const int token_start = chunk * ChunkSize;
    const int dimension = threadIdx.x;
    __shared__ float chunk_decay[ChunkSize];
    __shared__ float chunk_beta[ChunkSize];
    if (dimension == 0)
    {
        float sum = 0.0f;
#pragma unroll
        for (int offset = 0; offset < ChunkSize; ++offset)
        {
            const int token = token_start + offset;
            if (token < sequence_length)
            {
                const int64_t scalar_index = static_cast<int64_t>(head_batch) * padded_length + token;
                sum += log_decay[scalar_index];
                chunk_decay[offset] = sum;
                chunk_beta[offset] = beta[scalar_index];
                cumulative_decay[scalar_index] = sum;
            }
            else
            {
                chunk_decay[offset] = 0.0f;
                chunk_beta[offset] = 0.0f;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int offset = 0; offset < ChunkSize; ++offset)
    {
        const int token = token_start + offset;
        const int64_t index =
            (static_cast<int64_t>(head_batch) * padded_length + token) * HeadDimension + dimension;
        if (token < sequence_length)
        {
            const float key_value = CudaScalar<scalar_t>::to_float(key[index]);
            const float beta_value_float = chunk_beta[offset];
            key_beta_exp[index] =
                CudaScalar<scalar_t>::from_float(key_value * beta_value_float * expf(chunk_decay[offset]));
            key_exp_negative[index] = CudaScalar<scalar_t>::from_float(key_value * beta_value_float);
            beta_value[index] = CudaScalar<scalar_t>::from_float(
                CudaScalar<scalar_t>::to_float(value[index]) * beta_value_float);
        }
        else
        {
            key_beta_exp[index] = CudaScalar<scalar_t>::from_float(0.0f);
            key_exp_negative[index] = CudaScalar<scalar_t>::from_float(0.0f);
            beta_value[index] = CudaScalar<scalar_t>::from_float(0.0f);
        }
    }
}

template <int ChunkSize>
__global__ void gdn_scale_recurrence_matrix_kernel(float* matrix, const float* cumulative_decay,
                                                    int sequence_length, int padded_length, int matrix_count)
{
    const int matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) return;
    const int chunks_per_head = padded_length / ChunkSize;
    const int head_batch = matrix_index / chunks_per_head;
    const int chunk = matrix_index % chunks_per_head;
    const int token_start = chunk * ChunkSize;
    for (int index = threadIdx.x; index < ChunkSize * ChunkSize; index += blockDim.x)
    {
        const int row = index / ChunkSize;
        const int column = index % ChunkSize;
        const int token = token_start + row;
        const int key_token = token_start + column;
        float value = 0.0f;
        if (token < sequence_length && key_token < token)
        {
            const float row_decay = cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + token];
            const float column_decay =
                cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + key_token];
            value = matrix[static_cast<int64_t>(matrix_index) * ChunkSize * ChunkSize + index] *
                    expf(row_decay - column_decay);
        }
        matrix[static_cast<int64_t>(matrix_index) * ChunkSize * ChunkSize + index] = value;
    }
}

template <typename scalar_t, int ChunkSize>
__global__ void gdn_solve_lower_kernel(const float* matrix, scalar_t* inverse, int matrix_count)
{
    const int matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) return;
    __shared__ float lower[ChunkSize * ChunkSize];
    __shared__ float solved[ChunkSize * ChunkSize];
    for (int index = threadIdx.x; index < ChunkSize * ChunkSize; index += blockDim.x)
    {
        const int row = index / ChunkSize;
        const int column = index % ChunkSize;
        lower[index] = row > column ? matrix[static_cast<int64_t>(matrix_index) * ChunkSize * ChunkSize + index]
                                    : 0.0f;
        solved[index] = row == column ? 1.0f : 0.0f;
    }
    __syncthreads();
    const int column = threadIdx.x;
    if (column < ChunkSize)
    {
        for (int row = 1; row < ChunkSize; ++row)
        {
            if (column < row)
            {
                float value = -lower[row * ChunkSize + column];
                for (int inner = column + 1; inner < row; ++inner)
                    value -= lower[row * ChunkSize + inner] * solved[inner * ChunkSize + column];
                solved[row * ChunkSize + column] = value;
            }
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < ChunkSize * ChunkSize; index += blockDim.x)
        inverse[static_cast<int64_t>(matrix_index) * ChunkSize * ChunkSize + index] =
            CudaScalar<scalar_t>::from_float(solved[index]);
}

template <typename scalar_t, int ChunkSize>
__global__ void gdn_scale_solve_lower_kernel(const float* matrix, const float* cumulative_decay,
                                             scalar_t* inverse, int sequence_length, int padded_length,
                                             int matrix_count)
{
    const int matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) return;
    const int chunks_per_head = padded_length / ChunkSize;
    const int head_batch = matrix_index / chunks_per_head;
    const int chunk = matrix_index % chunks_per_head;
    const int token_start = chunk * ChunkSize;

    __shared__ float lower[ChunkSize * ChunkSize];
    __shared__ float solved[ChunkSize * ChunkSize];
    for (int index = threadIdx.x; index < ChunkSize * ChunkSize; index += blockDim.x)
    {
        const int row = index / ChunkSize;
        const int column = index % ChunkSize;
        const int token = token_start + row;
        const int key_token = token_start + column;
        float value = 0.0f;
        if (token < sequence_length && key_token < token)
        {
            const float row_decay = cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + token];
            const float column_decay =
                cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + key_token];
            value = matrix[static_cast<int64_t>(matrix_index) * ChunkSize * ChunkSize + index] *
                    expf(row_decay - column_decay);
        }
        lower[index] = row > column ? value : 0.0f;
        solved[index] = row == column ? 1.0f : 0.0f;
    }
    __syncthreads();

    const int column = threadIdx.x;
    if (column < ChunkSize)
    {
        for (int row = 1; row < ChunkSize; ++row)
        {
            if (column < row)
            {
                float value = -lower[row * ChunkSize + column];
                for (int inner = column + 1; inner < row; ++inner)
                    value -= lower[row * ChunkSize + inner] * solved[inner * ChunkSize + column];
                solved[row * ChunkSize + column] = value;
            }
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < ChunkSize * ChunkSize; index += blockDim.x)
        inverse[static_cast<int64_t>(matrix_index) * ChunkSize * ChunkSize + index] =
            CudaScalar<scalar_t>::from_float(solved[index]);
}

template <int HeadDimension, int VBlock, int ChunkSize, int WarpCount>
__global__ void gdn_chunk_delta_h_cute_kernel(const __nv_bfloat16* w, const __nv_bfloat16* key,
                                              const __nv_bfloat16* u, const float* cumulative_decay,
                                              __nv_bfloat16* chunk_states, __nv_bfloat16* new_value,
                                              __nv_bfloat16* scaled_value, float* working_state,
                                              int sequence_length, int padded_length, int chunks_per_head)
{
    using namespace cute;
    constexpr int state_elements = HeadDimension * VBlock;
    constexpr int chunk_vector_elements = ChunkSize * HeadDimension;
    constexpr int thread_count = WarpCount * warp_size;
    const int head_batch = blockIdx.x;
    const int v_tile = blockIdx.y * VBlock;
    const int tid = threadIdx.x;

    extern __shared__ unsigned char shared_raw[];
    __nv_bfloat16* h_shared = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    __nv_bfloat16* w_shared = h_shared + state_elements;
    float* projected = reinterpret_cast<float*>(w_shared + chunk_vector_elements);
    __nv_bfloat16* scaled_shared = reinterpret_cast<__nv_bfloat16*>(projected + ChunkSize * VBlock);
    float* decay_shared = reinterpret_cast<float*>(scaled_shared + ChunkSize * VBlock);

    // Swizzled K-major tiles let the 128-bit LDSM copy atoms feed the MMA directly.
    auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
    auto sW = make_tensor(make_smem_ptr(w_shared),
                          tile_to_shape(swizzle_atom, make_shape(Int<ChunkSize>{}, Int<HeadDimension>{})));
    // GEMM1 consumes w_shared before GEMM2 loads the transposed key into the same tile.
    auto sKey = make_tensor(make_smem_ptr(w_shared),
                            tile_to_shape(swizzle_atom, make_shape(Int<HeadDimension>{}, Int<ChunkSize>{})));
    auto sScaled = make_tensor(make_smem_ptr(scaled_shared),
                               tile_to_shape(swizzle_atom, make_shape(Int<VBlock>{}, Int<ChunkSize>{})));
    auto sH = make_tensor(make_smem_ptr(h_shared),
                          make_layout(make_shape(Int<VBlock>{}, Int<HeadDimension>{}),
                                      make_stride(Int<HeadDimension>{}, Int<1>{})));
    auto sHC = make_tensor(make_smem_ptr(h_shared),
                           make_layout(make_shape(Int<HeadDimension>{}, Int<VBlock>{}),
                                       make_stride(Int<1>{}, Int<HeadDimension>{})));
    auto sProj = make_tensor(make_smem_ptr(projected),
                             make_layout(make_shape(Int<ChunkSize>{}, Int<VBlock>{}),
                                         make_stride(Int<VBlock>{}, Int<1>{})));
    auto sDecay = make_tensor(make_smem_ptr(decay_shared),
                              make_layout(make_shape(Int<ChunkSize>{}), make_stride(Int<1>{})));

    using Mma = SM80_16x8x16_F32BF16BF16F32_TN;
    TiledMMA mma = make_tiled_mma(Mma{}, Layout<Shape<_2, _4>>{}, Tile<_32, _32, _16>{});
    ThrMMA thr_mma = mma.get_slice(tid);

    auto tCgHC = thr_mma.partition_C(sHC);
    auto tCrH = thr_mma.make_fragment_C(tCgHC);
    auto tCgP = thr_mma.partition_C(sProj);
    auto tCrP = thr_mma.make_fragment_C(tCgP);

    TiledCopy s2r_w = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>{}, mma);
    TiledCopy s2r_key = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>{}, mma);
    TiledCopy s2r_scaled = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, __nv_bfloat16>{}, mma);
    TiledCopy s2r_h = make_tiled_copy_B(Copy_Atom<UniversalCopy<__nv_bfloat16>, __nv_bfloat16>{}, mma);
    auto thr_w = s2r_w.get_slice(tid);
    auto thr_key = s2r_key.get_slice(tid);
    auto thr_scaled = s2r_scaled.get_slice(tid);
    auto thr_h = s2r_h.get_slice(tid);
    auto tXsW = thr_w.partition_S(sW);
    auto tXsKey = thr_key.partition_S(sKey);
    auto tXsScaled = thr_scaled.partition_S(sScaled);
    auto tXsH = thr_h.partition_S(sH);
    auto tCrAW = thr_mma.partition_fragment_A(sW);
    auto tCrAKey = thr_mma.partition_fragment_A(sKey);
    auto tCrBScaled = thr_mma.partition_fragment_B(sScaled);
    auto tCrBH = thr_mma.partition_fragment_B(sH);
    auto tXrW = thr_w.retile_D(tCrAW);
    auto tXrKey = thr_key.retile_D(tCrAKey);
    auto tXrScaled = thr_scaled.retile_D(tCrBScaled);
    auto tXrH = thr_h.retile_D(tCrBH);

    const __nv_bfloat16* w_base = w + static_cast<int64_t>(head_batch) * padded_length * HeadDimension;
    const __nv_bfloat16* key_base = key + static_cast<int64_t>(head_batch) * padded_length * HeadDimension;
    const __nv_bfloat16* u_base = u + static_cast<int64_t>(head_batch) * padded_length * HeadDimension;
    const float* decay_base = cumulative_decay + static_cast<int64_t>(head_batch) * padded_length;
    __nv_bfloat16* chunk_state_base =
        chunk_states + static_cast<int64_t>(head_batch) * chunks_per_head * HeadDimension * HeadDimension;
    __nv_bfloat16* new_value_base = new_value + static_cast<int64_t>(head_batch) * padded_length * HeadDimension;
    __nv_bfloat16* scaled_value_base =
        scaled_value + static_cast<int64_t>(head_batch) * padded_length * HeadDimension;

    const float* state_source = working_state + static_cast<int64_t>(head_batch) * HeadDimension * HeadDimension;
    for (int index = tid; index < state_elements; index += thread_count)
    {
        const int value_dimension = index / HeadDimension;
        const int key_dimension = index % HeadDimension;
        sH(value_dimension, key_dimension) = CudaScalar<__nv_bfloat16>::from_float(
            state_source[key_dimension * HeadDimension + v_tile + value_dimension]);
    }
    __syncthreads();
    copy(tCgHC, tCrH);

    for (int chunk = 0; chunk < chunks_per_head; ++chunk)
    {
        const int token_start = chunk * ChunkSize;
        const int last_token = token_start + ChunkSize - 1 < sequence_length
                                   ? token_start + ChunkSize - 1
                                   : sequence_length - 1;
        const float last_decay = decay_base[last_token];

        copy(tCrH, tCgHC);
        __syncthreads();

        __nv_bfloat16* chunk_states_out =
            chunk_state_base + static_cast<int64_t>(chunk) * HeadDimension * HeadDimension;
        for (int index = tid; index < state_elements; index += thread_count)
        {
            const int key_dimension = index / VBlock;
            const int value_dimension = index % VBlock;
            chunk_states_out[key_dimension * HeadDimension + v_tile + value_dimension] =
                sH(value_dimension, key_dimension);
        }

        const __nv_bfloat16* w_chunk = w_base + chunk * chunk_vector_elements;
        const __nv_bfloat16* key_chunk = key_base + chunk * chunk_vector_elements;
        constexpr int chunk_vectors = chunk_vector_elements / 8;
        for (int index = tid; index < chunk_vectors; index += thread_count)
        {
            const uint4 w_value = reinterpret_cast<const uint4*>(w_chunk)[index];
            const int token = (index * 8) / HeadDimension;
            const int key_dimension = (index * 8) % HeadDimension;
            *reinterpret_cast<uint4*>(&sW(token, key_dimension)) = w_value;
        }
        if (tid < ChunkSize)
        {
            const int token = token_start + tid;
            sDecay(tid) =
                token < sequence_length ? expf(last_decay - decay_base[token]) : 0.0f;
        }
        __syncthreads();

        clear(tCrP);
        copy(s2r_w, tXsW, tXrW);
        copy(s2r_h, tXsH, tXrH);
        gemm(mma, tCrAW, tCrBH, tCrP);
        copy(tCrP, tCgP);
        __syncthreads();

        for (int index = tid; index < chunk_vectors; index += thread_count)
        {
            const uint4 key_value = reinterpret_cast<const uint4*>(key_chunk)[index];
            const int token = (index * 8) / HeadDimension;
            const int key_dimension = (index * 8) % HeadDimension;
            const __nv_bfloat16* key_elements = reinterpret_cast<const __nv_bfloat16*>(&key_value);
#pragma unroll
            for (int i = 0; i < 8; ++i) sKey(key_dimension + i, token) = key_elements[i];
        }

        for (int index = tid; index < ChunkSize * VBlock; index += thread_count)
        {
            const int token = token_start + index / VBlock;
            const int value_dimension = index % VBlock;
            const int64_t packed =
                static_cast<int64_t>(token) * HeadDimension + v_tile + value_dimension;
            if (token < sequence_length)
            {
                const float value = CudaScalar<__nv_bfloat16>::to_float(u_base[packed]) - projected[index];
                const __nv_bfloat16 rounded = CudaScalar<__nv_bfloat16>::from_float(value);
                const __nv_bfloat16 scaled = CudaScalar<__nv_bfloat16>::from_float(
                    CudaScalar<__nv_bfloat16>::to_float(rounded) * sDecay(index / VBlock));
                new_value_base[packed] = rounded;
                scaled_value_base[packed] = scaled;
                sScaled(value_dimension, index / VBlock) = scaled;
            }
            else
            {
                const __nv_bfloat16 zero = CudaScalar<__nv_bfloat16>::from_float(0.0f);
                new_value_base[packed] = zero;
                scaled_value_base[packed] = zero;
                sScaled(value_dimension, index / VBlock) = zero;
            }
        }
        __syncthreads();

        const float state_decay = expf(last_decay);
        for (int index = 0; index < tCrH.size(); ++index) tCrH(index) *= state_decay;

        copy(s2r_key, tXsKey, tXrKey);
        copy(s2r_scaled, tXsScaled, tXrScaled);
        gemm(mma, tCrAKey, tCrBScaled, tCrH);
    }

    copy(tCrH, tCgHC);
    __syncthreads();
    float* state_destination = working_state + static_cast<int64_t>(head_batch) * HeadDimension * HeadDimension;
    for (int index = tid; index < state_elements; index += thread_count)
    {
        const int key_dimension = index / VBlock;
        const int value_dimension = index % VBlock;
        state_destination[key_dimension * HeadDimension + v_tile + value_dimension] =
            CudaScalar<__nv_bfloat16>::to_float(sH(value_dimension, key_dimension));
    }
}

template <typename scalar_t, typename state_scalar_t, int HeadDimension, int ChunkSize>
__global__ void gdn_gather_state_kernel(const state_scalar_t* states, const int* state_slots,
                                        const int* context_lengths, float* working_state, int batch, int head_count)
{
    const int head_batch = blockIdx.x;
    const int batch_index = head_batch / head_count;
    const int head = head_batch % head_count;
    const int slot = state_slots[batch_index];
    const state_scalar_t* source = states +
        ((static_cast<int64_t>(slot) * head_count + head) * HeadDimension * HeadDimension);
    float* destination = working_state + static_cast<int64_t>(head_batch) * HeadDimension * HeadDimension;
    const bool clear = !context_lengths || context_lengths[batch_index] == 0;
    for (int index = blockIdx.y * blockDim.x + threadIdx.x; index < HeadDimension * HeadDimension;
         index += blockDim.x * gridDim.y)
        destination[index] =
            clear ? 0.0f : CudaScalar<state_scalar_t>::to_float(source[index]);
}

template <typename scalar_t, int HeadDimension, int ChunkSize>
__global__ void gdn_prepare_state_update_kernel(const float* update, const scalar_t* projected_value,
                                                 const float* cumulative_decay, scalar_t* new_value,
                                                 scalar_t* scaled_value, float* working_state, int chunk,
                                                 int sequence_length, int padded_length, int head_batches)
{
    const int head_batch = blockIdx.x;
    if (head_batch >= head_batches) return;
    const int token_start = chunk * ChunkSize;
    const int last_token = min(token_start + ChunkSize, sequence_length) - 1;
    const float last_decay = cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + last_token];
    __shared__ float state_decay;
    if (threadIdx.x == 0) state_decay = expf(last_decay);
    __syncthreads();
    for (int index = blockIdx.y * blockDim.x + threadIdx.x; index < HeadDimension * HeadDimension;
         index += blockDim.x * gridDim.y)
        working_state[static_cast<int64_t>(head_batch) * HeadDimension * HeadDimension + index] *= state_decay;
    for (int index = blockIdx.y * blockDim.x + threadIdx.x; index < ChunkSize * HeadDimension;
         index += blockDim.x * gridDim.y)
    {
        const int offset = index / HeadDimension;
        const int dimension = index % HeadDimension;
        const int token = token_start + offset;
        const int64_t packed =
            (static_cast<int64_t>(head_batch) * padded_length + token) * HeadDimension + dimension;
        if (token < sequence_length)
        {
            const float value = CudaScalar<scalar_t>::to_float(projected_value[packed]) -
                                update[packed];
            const scalar_t rounded = CudaScalar<scalar_t>::from_float(value);
            new_value[packed] = rounded;
            const float token_decay = cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + token];
            scaled_value[packed] = CudaScalar<scalar_t>::from_float(
                CudaScalar<scalar_t>::to_float(rounded) * expf(last_decay - token_decay));
        }
        else
        {
            new_value[packed] = CudaScalar<scalar_t>::from_float(0.0f);
            scaled_value[packed] = CudaScalar<scalar_t>::from_float(0.0f);
        }
    }
}

template <typename scalar_t, int HeadDimension>
__global__ void gdn_save_state_kernel(const float* working_state, scalar_t* chunk_states, int chunk,
                                      int chunks_per_head, int head_batches)
{
    const int head_batch = blockIdx.x;
    if (head_batch >= head_batches) return;
    const float* source = working_state + static_cast<int64_t>(head_batch) * HeadDimension * HeadDimension;
    scalar_t* destination = chunk_states +
        (static_cast<int64_t>(head_batch) * chunks_per_head + chunk) * HeadDimension * HeadDimension;
    for (int index = blockIdx.y * blockDim.x + threadIdx.x; index < HeadDimension * HeadDimension;
         index += blockDim.x * gridDim.y)
        destination[index] = CudaScalar<scalar_t>::from_float(source[index]);
}

template <typename scalar_t, int ChunkSize>
__global__ void gdn_scale_chunk_output_kernel(float* output, const float* scores, scalar_t* scaled_scores,
                                               const float* cumulative_decay, int sequence_length,
                                               int padded_length, int matrix_count)
{
    const int matrix = blockIdx.x;
    if (matrix >= matrix_count) return;
    const int chunks_per_head = padded_length / ChunkSize;
    const int head_batch = matrix / chunks_per_head;
    const int chunk = matrix % chunks_per_head;
    const int token_start = chunk * ChunkSize;
    for (int index = threadIdx.x; index < ChunkSize * ChunkSize; index += blockDim.x)
    {
        const int row = index / ChunkSize;
        const int column = index % ChunkSize;
        const int token = token_start + row;
        const int key_token = token_start + column;
        const int64_t score_index = static_cast<int64_t>(matrix) * ChunkSize * ChunkSize + index;
        float value = 0.0f;
        if (token < sequence_length && key_token <= token)
        {
            const float row_decay = cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + token];
            const float column_decay =
                cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + key_token];
            value = scores[score_index] * expf(row_decay - column_decay);
        }
        scaled_scores[score_index] = CudaScalar<scalar_t>::from_float(value);
    }
    for (int index = threadIdx.x; index < ChunkSize * 128; index += blockDim.x)
    {
        const int row = index / 128;
        const int token = token_start + row;
        if (token < sequence_length)
        {
            const int64_t output_index =
                (static_cast<int64_t>(head_batch) * padded_length + token) * 128 + index % 128;
            const float decay_value = cumulative_decay[static_cast<int64_t>(head_batch) * padded_length + token];
            output[output_index] *= expf(decay_value);
        }
    }
}

template <typename scalar_t, int HeadDimension>
__global__ void gdn_unpack_output_kernel(const float* packed_output, const scalar_t* gate,
                                          const float* norm_weight, scalar_t* output, int sequence_length,
                                          int padded_length, int head_count, int gate_stride, float epsilon)
{
    const int row = blockIdx.x;
    const int head = row % head_count;
    const int token_index = row / head_count;
    const int batch = token_index / sequence_length;
    const int token = token_index % sequence_length;
    const int dimension = threadIdx.x;
    const int64_t packed_index =
        ((static_cast<int64_t>(batch) * head_count + head) * padded_length + token) * HeadDimension + dimension;
    const float recurrent_value = CudaScalar<scalar_t>::to_float(
        CudaScalar<scalar_t>::from_float(packed_output[packed_index]));
    const float square_sum = block_sum<HeadDimension>(recurrent_value * recurrent_value);
    const float inverse_rms = rsqrtf(square_sum / static_cast<float>(HeadDimension) + epsilon);
    const int64_t output_index = static_cast<int64_t>(row) * HeadDimension + dimension;
    const int64_t gate_index = static_cast<int64_t>(token_index) * gate_stride + head * HeadDimension + dimension;
    const float gate_value = CudaScalar<scalar_t>::to_float(gate[gate_index]);
    const float silu_gate = gate_value / (1.0f + expf(-gate_value));
    const float normalized = CudaScalar<scalar_t>::to_float(
        CudaScalar<scalar_t>::from_float(recurrent_value * inverse_rms));
    output[output_index] =
        CudaScalar<scalar_t>::from_float(normalized * norm_weight[dimension] * silu_gate);
}

template <typename state_scalar_t, int HeadDimension>
__global__ void gdn_scatter_state_kernel(const float* working_state, state_scalar_t* states,
                                         const int* state_slots, int head_count, int head_batches)
{
    const int head_batch = blockIdx.x;
    if (head_batch >= head_batches) return;
    const int batch = head_batch / head_count;
    const int head = head_batch % head_count;
    const int slot = state_slots[batch];
    const float* source = working_state + static_cast<int64_t>(head_batch) * HeadDimension * HeadDimension;
    state_scalar_t* destination = states +
        ((static_cast<int64_t>(slot) * head_count + head) * HeadDimension * HeadDimension);
    for (int index = blockIdx.y * blockDim.x + threadIdx.x; index < HeadDimension * HeadDimension;
         index += blockDim.x * gridDim.y)
        destination[index] = CudaScalar<state_scalar_t>::from_float(source[index]);
}

template <typename scalar_t, typename state_scalar_t, int HeadDimension, int ChunkSize>
void gated_delta_net_chunk(const scalar_t* mixed_qkv, const scalar_t* gate, const scalar_t* decay,
                           const scalar_t* beta, const float* decay_log, const scalar_t* decay_bias,
                           const float* norm_weight, state_scalar_t* states, const int* state_slots,
                           const int* context_lengths, scalar_t* output, int batch, int sequence_length,
                           int head_count, int gate_stride, int scalar_stride,
                           GatedDeltaNetWorkspace::Impl& workspace, float epsilon,
                           const device::Context& context)
{
    const int padded_length = ((sequence_length + ChunkSize - 1) / ChunkSize) * ChunkSize;
    const int chunks_per_head = padded_length / ChunkSize;
    const int head_batches = batch * head_count;
    const int matrix_count = head_batches * chunks_per_head;
    const DType scalar_dtype = std::is_same_v<scalar_t, __nv_bfloat16> ? DType::BF16 : DType::F16;
    reserve_chunk_workspace<HeadDimension, ChunkSize>(workspace, batch, sequence_length, head_count, scalar_dtype,
                                                       context);
    Tensor& query = workspace.query;
    Tensor& key = workspace.key;
    Tensor& value = workspace.value;
    Tensor& log_decay = workspace.log_decay;
    Tensor& beta_values = workspace.beta_values;
    Tensor& key_beta_exp = workspace.key_beta_exp;
    Tensor& key_exp_negative = workspace.key_exp_negative;
    Tensor& beta_value = workspace.beta_value;
    Tensor& cumulative_decay = workspace.cumulative_decay;
    Tensor& matrix = workspace.matrix;
    Tensor& inverse = workspace.inverse;
    Tensor& w = workspace.w;
    Tensor& u = workspace.u;
    Tensor& chunk_states = workspace.chunk_states;
    Tensor& working_state = workspace.working_state;
    Tensor& projected = workspace.projected;
    Tensor& new_value = workspace.new_value;
    Tensor& scaled_value = workspace.scaled_value;
    Tensor& packed_output = workspace.packed_output;
    Tensor& scaled_scores = workspace.scaled_scores;

    cudaMemsetAsync(query.data(), 0, query.nbytes(), context.stream());
    cudaMemsetAsync(key.data(), 0, key.nbytes(), context.stream());
    cudaMemsetAsync(value.data(), 0, value.nbytes(), context.stream());
    const int token_rows = batch * sequence_length * head_count;
    static const bool scalar_pack = []
    {
        const char* backend = std::getenv("FIREFLY_QWEN35_GDN_PACK_BACKEND");
        return backend && std::strcmp(backend, "scalar") == 0;
    }();
    if (scalar_pack)
        gdn_pack_scalar_kernel<scalar_t, HeadDimension><<<token_rows, HeadDimension, 0, context.stream()>>>(
            mixed_qkv, decay, beta, decay_log, decay_bias, static_cast<scalar_t*>(query.data()),
            static_cast<scalar_t*>(key.data()), static_cast<scalar_t*>(value.data()),
            static_cast<float*>(log_decay.data()), static_cast<float*>(beta_values.data()), sequence_length,
            padded_length, head_count, scalar_stride);
    else
        gdn_pack_vector_kernel<scalar_t, HeadDimension><<<token_rows, warp_size, 0, context.stream()>>>(
            mixed_qkv, decay, beta, decay_log, decay_bias, static_cast<scalar_t*>(query.data()),
            static_cast<scalar_t*>(key.data()), static_cast<scalar_t*>(value.data()),
            static_cast<float*>(log_decay.data()), static_cast<float*>(beta_values.data()), sequence_length,
            padded_length, head_count, scalar_stride);
    gdn_chunk_factors_kernel<scalar_t, HeadDimension, ChunkSize><<<matrix_count, HeadDimension, 0, context.stream()>>>(
        static_cast<const scalar_t*>(key.data()), static_cast<const scalar_t*>(value.data()),
        static_cast<const float*>(log_decay.data()), static_cast<const float*>(beta_values.data()),
        static_cast<scalar_t*>(key_beta_exp.data()), static_cast<scalar_t*>(key_exp_negative.data()),
        static_cast<scalar_t*>(beta_value.data()), static_cast<float*>(cumulative_decay.data()), sequence_length,
        padded_length, chunks_per_head);

    constexpr int64_t chunk_vector_elements = ChunkSize * HeadDimension;
    constexpr int64_t chunk_matrix_elements = ChunkSize * ChunkSize;
    constexpr int64_t state_elements = HeadDimension * HeadDimension;
    batched_gemm(static_cast<const scalar_t*>(key_exp_negative.data()), static_cast<const scalar_t*>(key.data()),
                 matrix.data(), ChunkSize, ChunkSize,
                 HeadDimension, false, true, chunk_vector_elements, chunk_vector_elements, chunk_matrix_elements,
                 matrix_count, CUDA_R_32F, 1.0f, 0.0f, context.stream());
    constexpr int solve_threads = std::max(ChunkSize, warp_size);
    gdn_scale_solve_lower_kernel<scalar_t, ChunkSize><<<matrix_count, solve_threads, 0, context.stream()>>>(
        static_cast<const float*>(matrix.data()), static_cast<const float*>(cumulative_decay.data()),
        static_cast<scalar_t*>(inverse.data()), sequence_length, padded_length, matrix_count);
    batched_gemm(static_cast<const scalar_t*>(inverse.data()), static_cast<const scalar_t*>(key_beta_exp.data()),
                 w.data(), ChunkSize, HeadDimension, ChunkSize, false, false, chunk_matrix_elements,
                 chunk_vector_elements, chunk_vector_elements, matrix_count, cublas_data_type<scalar_t>(), 1.0f,
                 0.0f, context.stream());
    batched_gemm(static_cast<const scalar_t*>(inverse.data()), static_cast<const scalar_t*>(beta_value.data()),
                 u.data(), ChunkSize, HeadDimension, ChunkSize, false, false, chunk_matrix_elements,
                 chunk_vector_elements, chunk_vector_elements, matrix_count, cublas_data_type<scalar_t>(), 1.0f,
                 0.0f, context.stream());

    constexpr int state_tiles = 8;
    dim3 state_grid(head_batches, state_tiles);
    gdn_gather_state_kernel<scalar_t, state_scalar_t, HeadDimension, ChunkSize>
        <<<state_grid, 256, 0, context.stream()>>>(
        states, state_slots, context_lengths, static_cast<float*>(working_state.data()), batch, head_count);
    const char* prefill_backend = std::getenv("FIREFLY_QWEN35_GDN_PREFILL_BACKEND");
    const bool use_fused = std::is_same_v<scalar_t, __nv_bfloat16> &&
                           (!prefill_backend || std::strcmp(prefill_backend, "fused") == 0);
    auto run_chunk_recurrence = [&]()
    {
        for (int chunk = 0; chunk < chunks_per_head; ++chunk)
        {
            gdn_save_state_kernel<scalar_t, HeadDimension><<<state_grid, 256, 0, context.stream()>>>(
                static_cast<const float*>(working_state.data()), static_cast<scalar_t*>(chunk_states.data()),
                chunk, chunks_per_head, head_batches);
            batched_gemm(static_cast<const scalar_t*>(w.data()) + chunk * chunk_vector_elements,
                         static_cast<const scalar_t*>(chunk_states.data()) + chunk * state_elements,
                         static_cast<float*>(projected.data()) + chunk * chunk_vector_elements, ChunkSize,
                         HeadDimension, HeadDimension, false, false,
                         static_cast<int64_t>(padded_length) * HeadDimension,
                         static_cast<int64_t>(chunks_per_head) * state_elements,
                         static_cast<int64_t>(padded_length) * HeadDimension, head_batches, CUDA_R_32F, 1.0f,
                         0.0f, context.stream());
            gdn_prepare_state_update_kernel<scalar_t, HeadDimension, ChunkSize>
                <<<state_grid, 256, 0, context.stream()>>>(
                    static_cast<const float*>(projected.data()), static_cast<const scalar_t*>(u.data()),
                    static_cast<const float*>(cumulative_decay.data()), static_cast<scalar_t*>(new_value.data()),
                    static_cast<scalar_t*>(scaled_value.data()), static_cast<float*>(working_state.data()), chunk,
                    sequence_length, padded_length, head_batches);
            batched_gemm(static_cast<const scalar_t*>(key.data()) + chunk * chunk_vector_elements,
                         static_cast<const scalar_t*>(scaled_value.data()) + chunk * chunk_vector_elements,
                         working_state.data(), HeadDimension, HeadDimension, ChunkSize, true, false,
                         static_cast<int64_t>(padded_length) * HeadDimension,
                         static_cast<int64_t>(padded_length) * HeadDimension, state_elements, head_batches,
                         CUDA_R_32F, 1.0f, 1.0f, context.stream());
        }
    };
    if (use_fused)
    {
        constexpr int FusedChunkSize = 64;
        constexpr int WarpCount = 8;
        constexpr int VBlock = 32;
        constexpr int fused_state_bytes =
            (HeadDimension * VBlock + FusedChunkSize * HeadDimension + FusedChunkSize * VBlock) *
                sizeof(__nv_bfloat16) +
            FusedChunkSize * VBlock * sizeof(float) + FusedChunkSize * sizeof(float);
        static const bool fused_state_configured = []()
        {
            return cudaFuncSetAttribute(
                       gdn_chunk_delta_h_cute_kernel<HeadDimension, VBlock, FusedChunkSize, WarpCount>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        fused_state_bytes) == cudaSuccess;
        }();
        if (!fused_state_configured)
            throw std::runtime_error("linear attention fused GDN recurrence shared memory exceeds device limit");
        dim3 fused_grid(head_batches, HeadDimension / VBlock);
        gdn_chunk_delta_h_cute_kernel<HeadDimension, VBlock, FusedChunkSize, WarpCount>
            <<<fused_grid, WarpCount * warp_size, fused_state_bytes, context.stream()>>>(
                static_cast<const __nv_bfloat16*>(w.data()), static_cast<const __nv_bfloat16*>(key.data()),
                static_cast<const __nv_bfloat16*>(u.data()),
                static_cast<const float*>(cumulative_decay.data()),
                static_cast<__nv_bfloat16*>(chunk_states.data()),
                static_cast<__nv_bfloat16*>(new_value.data()),
                static_cast<__nv_bfloat16*>(scaled_value.data()), static_cast<float*>(working_state.data()),
                sequence_length, padded_length, chunks_per_head);
        check_launch("linear attention fused GDN recurrence");
    }
    if (!use_fused) run_chunk_recurrence();

    batched_gemm(static_cast<const scalar_t*>(query.data()), static_cast<const scalar_t*>(chunk_states.data()),
                 packed_output.data(), ChunkSize, HeadDimension, HeadDimension, false, false,
                 chunk_vector_elements, state_elements, chunk_vector_elements, matrix_count, CUDA_R_32F, 1.0f, 0.0f,
                 context.stream());
    batched_gemm(static_cast<const scalar_t*>(query.data()), static_cast<const scalar_t*>(key.data()), matrix.data(),
                 ChunkSize, ChunkSize, HeadDimension, false, true, chunk_vector_elements, chunk_vector_elements,
                 chunk_matrix_elements, matrix_count, CUDA_R_32F, 1.0f, 0.0f, context.stream());
    gdn_scale_chunk_output_kernel<scalar_t, ChunkSize><<<matrix_count, 256, 0, context.stream()>>>(
        static_cast<float*>(packed_output.data()), static_cast<const float*>(matrix.data()),
        static_cast<scalar_t*>(scaled_scores.data()), static_cast<const float*>(cumulative_decay.data()),
        sequence_length, padded_length, matrix_count);
    batched_gemm(static_cast<const scalar_t*>(scaled_scores.data()), static_cast<const scalar_t*>(new_value.data()),
                 packed_output.data(), ChunkSize, HeadDimension, ChunkSize, false, false, chunk_matrix_elements,
                 chunk_vector_elements, chunk_vector_elements, matrix_count, CUDA_R_32F, 1.0f, 1.0f,
                 context.stream());
    gdn_unpack_output_kernel<scalar_t, HeadDimension><<<token_rows, HeadDimension, 0, context.stream()>>>(
        static_cast<const float*>(packed_output.data()), gate, norm_weight, output, sequence_length, padded_length,
        head_count, gate_stride, epsilon);
    gdn_scatter_state_kernel<state_scalar_t, HeadDimension><<<state_grid, 256, 0, context.stream()>>>(
        static_cast<const float*>(working_state.data()), states, state_slots, head_count, head_batches);
}

template <typename Function>
void dispatch_half_type(DType dtype, Function&& function)
{
    require_float16_or_bfloat16(dtype, "linear attention kernel");
    if (dtype == DType::BF16) function.template operator()<__nv_bfloat16>();
    else function.template operator()<half>();
}

template <typename Function>
void dispatch_state_type(DType dtype, Function&& function)
{
    if (dtype == DType::BF16) function.template operator()<__nv_bfloat16>();
    else if (dtype == DType::F16) function.template operator()<half>();
    else if (dtype == DType::F32) function.template operator()<float>();
    else throw std::runtime_error("linear attention recurrent state dtype must be F16, BF16, or F32");
}

void check_launch(const char* operation)
{
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
}  // namespace

GatedDeltaNetWorkspace::GatedDeltaNetWorkspace() : impl_(std::make_unique<Impl>()) {}
GatedDeltaNetWorkspace::~GatedDeltaNetWorkspace() = default;
GatedDeltaNetWorkspace::GatedDeltaNetWorkspace(GatedDeltaNetWorkspace&&) noexcept = default;
GatedDeltaNetWorkspace& GatedDeltaNetWorkspace::operator=(GatedDeltaNetWorkspace&&) noexcept = default;


void causal_convolution(const Tensor& projected_qkv, const Tensor& weight, Tensor& convolution_state,
                        const int* state_slots, const int* context_lengths, Tensor& output,
                        const device::Context& context)
{
    if (!state_slots) throw std::runtime_error("linear attention causal convolution requires state slots");
    const int batch = projected_qkv.shape()[0];
    const int sequence_length = projected_qkv.shape()[1];
    const int channels = projected_qkv.shape()[2];
    const int input_stride = projected_qkv.strides()[1];
    const int kernel_size = weight.shape().back();
    dispatch_half_type(projected_qkv.dtype(), [&]<typename scalar_t>()
    {
        if (sequence_length == 1)
        {
            dim3 grid((channels + 255) / 256, batch);
            causal_convolution_kernel<scalar_t><<<grid, 256, 0, context.stream()>>>(
                static_cast<const scalar_t*>(projected_qkv.data()), static_cast<const scalar_t*>(weight.data()),
                static_cast<scalar_t*>(convolution_state.data()), state_slots, context_lengths,
                static_cast<scalar_t*>(output.data()), sequence_length, channels, input_stride, kernel_size);
            return;
        }
        constexpr int token_tile = 4;
        dim3 prefill_grid((sequence_length + token_tile - 1) / token_tile, (channels + 255) / 256, batch);
        causal_convolution_prefill_tiled_kernel<scalar_t, token_tile><<<prefill_grid, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(projected_qkv.data()), static_cast<const scalar_t*>(weight.data()),
            static_cast<const scalar_t*>(convolution_state.data()), state_slots, context_lengths,
            static_cast<scalar_t*>(output.data()), sequence_length, channels, input_stride, kernel_size);
        dim3 state_grid((channels + 255) / 256, batch);
        causal_convolution_save_state_kernel<scalar_t><<<state_grid, 256, 0, context.stream()>>>(
            static_cast<const scalar_t*>(projected_qkv.data()), static_cast<scalar_t*>(convolution_state.data()),
            state_slots, context_lengths, sequence_length, channels, input_stride, kernel_size);
    });
    check_launch("linear attention causal convolution");
}

void gated_delta_net(const Tensor& mixed_qkv, const Tensor& gate, const Tensor& decay, const Tensor& beta,
                     const Tensor& decay_log, const Tensor& decay_bias, const Tensor& norm_weight,
                     Tensor& recurrent_state, const int* state_slots, const int* context_lengths, Tensor& output,
                     GatedDeltaNetWorkspace* workspace, double epsilon, const device::Context& context)
{
    if (!state_slots || decay_log.dtype() != DType::F32 || norm_weight.dtype() != DType::F32)
        throw std::runtime_error("invalid linear attention Gated DeltaNet state or parameter dtype");
    const int batch = mixed_qkv.shape()[0];
    const int sequence_length = mixed_qkv.shape()[1];
    const int head_count = decay.shape()[2];
    const int gate_stride = gate.strides()[1];
    const int decay_stride = decay.strides()[1];
    const int beta_stride = beta.strides()[1];
    if (decay_stride != beta_stride) throw std::runtime_error("linear attention Gated DeltaNet scalar stride mismatch");
    constexpr int head_dimension = 128;
    if (mixed_qkv.shape()[2] != head_count * head_dimension * 3 || output.shape()[2] != head_count * head_dimension)
        throw std::runtime_error("unsupported linear attention Gated DeltaNet shape");
    dispatch_half_type(mixed_qkv.dtype(), [&]<typename scalar_t>()
    {
        dispatch_state_type(recurrent_state.dtype(), [&]<typename state_scalar_t>()
        {
            if (sequence_length == 1)
            {
                constexpr int decode_state_bytes = head_dimension * head_dimension * sizeof(float);
                static const bool decode_state_configured = []()
                {
                    return cudaFuncSetAttribute(
                               gated_delta_net_kernel<scalar_t, state_scalar_t, head_dimension, true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               decode_state_bytes) == cudaSuccess;
                }();
                if (!decode_state_configured)
                    throw std::runtime_error("linear attention GDN decode state shared memory exceeds device limit");
                dim3 grid(batch, head_count);
                gated_delta_net_kernel<scalar_t, state_scalar_t, head_dimension, true>
                    <<<grid, head_dimension, decode_state_bytes, context.stream()>>>(
                        static_cast<const scalar_t*>(mixed_qkv.data()), static_cast<const scalar_t*>(gate.data()),
                        static_cast<const scalar_t*>(decay.data()), static_cast<const scalar_t*>(beta.data()),
                        static_cast<const float*>(decay_log.data()), static_cast<const scalar_t*>(decay_bias.data()),
                        static_cast<const float*>(norm_weight.data()),
                        static_cast<state_scalar_t*>(recurrent_state.data()), state_slots, context_lengths,
                        static_cast<scalar_t*>(output.data()), sequence_length, head_count, gate_stride,
                        decay_stride, static_cast<float>(epsilon));
                return;
            }
            if (!workspace || !workspace->impl_) throw std::runtime_error("linear attention GDN prefill requires workspace");
            const char* prefill_backend = std::getenv("FIREFLY_QWEN35_GDN_PREFILL_BACKEND");
            if (!prefill_backend || std::strcmp(prefill_backend, "fused") == 0 ||
                std::strcmp(prefill_backend, "chunk64") == 0 ||
                std::strcmp(prefill_backend, "chunk32") == 0 || std::strcmp(prefill_backend, "chunk16") == 0)
            {
                if (prefill_backend && std::strcmp(prefill_backend, "chunk32") == 0)
                    gated_delta_net_chunk<scalar_t, state_scalar_t, head_dimension, 32>(
                        static_cast<const scalar_t*>(mixed_qkv.data()), static_cast<const scalar_t*>(gate.data()),
                        static_cast<const scalar_t*>(decay.data()), static_cast<const scalar_t*>(beta.data()),
                        static_cast<const float*>(decay_log.data()), static_cast<const scalar_t*>(decay_bias.data()),
                        static_cast<const float*>(norm_weight.data()),
                        static_cast<state_scalar_t*>(recurrent_state.data()), state_slots, context_lengths,
                        static_cast<scalar_t*>(output.data()), batch, sequence_length, head_count, gate_stride,
                        decay_stride, *workspace->impl_, static_cast<float>(epsilon), context);
                else if (prefill_backend && std::strcmp(prefill_backend, "chunk16") == 0)
                    gated_delta_net_chunk<scalar_t, state_scalar_t, head_dimension, 16>(
                        static_cast<const scalar_t*>(mixed_qkv.data()), static_cast<const scalar_t*>(gate.data()),
                        static_cast<const scalar_t*>(decay.data()), static_cast<const scalar_t*>(beta.data()),
                        static_cast<const float*>(decay_log.data()), static_cast<const scalar_t*>(decay_bias.data()),
                        static_cast<const float*>(norm_weight.data()),
                        static_cast<state_scalar_t*>(recurrent_state.data()), state_slots, context_lengths,
                        static_cast<scalar_t*>(output.data()), batch, sequence_length, head_count, gate_stride,
                        decay_stride, *workspace->impl_, static_cast<float>(epsilon), context);
                else
                    gated_delta_net_chunk<scalar_t, state_scalar_t, head_dimension, 64>(
                        static_cast<const scalar_t*>(mixed_qkv.data()), static_cast<const scalar_t*>(gate.data()),
                        static_cast<const scalar_t*>(decay.data()), static_cast<const scalar_t*>(beta.data()),
                        static_cast<const float*>(decay_log.data()), static_cast<const scalar_t*>(decay_bias.data()),
                        static_cast<const float*>(norm_weight.data()),
                        static_cast<state_scalar_t*>(recurrent_state.data()), state_slots, context_lengths,
                        static_cast<scalar_t*>(output.data()), batch, sequence_length, head_count, gate_stride,
                        decay_stride, *workspace->impl_, static_cast<float>(epsilon), context);
                return;
            }
            const bool use_shared_state = prefill_backend && std::strcmp(prefill_backend, "shared") == 0;
            constexpr int shared_state_bytes = head_dimension * head_dimension * sizeof(float);
            static const bool shared_state_supported = [=]
            {
                int device = 0;
                int max_shared_bytes = 0;
                if (cudaGetDevice(&device) != cudaSuccess ||
                    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
                        cudaSuccess ||
                    max_shared_bytes < shared_state_bytes + 2048)
                    return false;
                return cudaFuncSetAttribute(gated_delta_net_kernel<scalar_t, state_scalar_t, head_dimension, true>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            shared_state_bytes) == cudaSuccess;
            }();
            dim3 grid(batch, head_count);
            if (use_shared_state && shared_state_supported)
            {
                gated_delta_net_kernel<scalar_t, state_scalar_t, head_dimension, true>
                    <<<grid, head_dimension, shared_state_bytes, context.stream()>>>(
                        static_cast<const scalar_t*>(mixed_qkv.data()), static_cast<const scalar_t*>(gate.data()),
                        static_cast<const scalar_t*>(decay.data()), static_cast<const scalar_t*>(beta.data()),
                        static_cast<const float*>(decay_log.data()), static_cast<const scalar_t*>(decay_bias.data()),
                        static_cast<const float*>(norm_weight.data()),
                        static_cast<state_scalar_t*>(recurrent_state.data()), state_slots, context_lengths,
                        static_cast<scalar_t*>(output.data()), sequence_length, head_count, gate_stride,
                        decay_stride, static_cast<float>(epsilon));
            }
            else
            {
                gated_delta_net_kernel<scalar_t, state_scalar_t, head_dimension, true>
                    <<<grid, head_dimension, shared_state_bytes, context.stream()>>>(
                        static_cast<const scalar_t*>(mixed_qkv.data()), static_cast<const scalar_t*>(gate.data()),
                        static_cast<const scalar_t*>(decay.data()), static_cast<const scalar_t*>(beta.data()),
                        static_cast<const float*>(decay_log.data()), static_cast<const scalar_t*>(decay_bias.data()),
                        static_cast<const float*>(norm_weight.data()),
                        static_cast<state_scalar_t*>(recurrent_state.data()), state_slots, context_lengths,
                        static_cast<scalar_t*>(output.data()), sequence_length, head_count, gate_stride,
                        decay_stride, static_cast<float>(epsilon));
            }
        });
    });
    check_launch("linear attention Gated DeltaNet");
}

}  // namespace firefly::kernels::linear_attention
