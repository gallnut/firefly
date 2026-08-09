firefly_add_object_module(firefly_kernels
    transformer/embedding.cu
    transformer/rms_norm.cu
    transformer/qk_rms_norm_rope.cu
    transformer/swiglu.cu
    transformer/rope.cu
    transformer/linear.cu
    transformer/residual.cu
    attention/attention.cc
    attention/decode_workspace.cc
    attention/contiguous_attention.cu
    attention/paged_attention.cc
    attention/paged_prefill.cu
    attention/paged_decode.cu
    attention/quantized_paged_attention.cu
    attention/flashinfer.cu
    attention/hybrid_attention.cu
    attention/ragged_attention.cu
    linear_attention/gated_delta_net.cu
    cache/kv_cache.cu
    sampling/argmax.cu
)
target_link_libraries(firefly_kernels
    PUBLIC CUDA::cudart CUDA::cublas
    PRIVATE nvidia::cutlass::cutlass
)
if(FIREFLY_HAS_FLASHINFER)
    target_compile_definitions(firefly_kernels PRIVATE FIREFLY_USE_FLASHINFER)
    target_include_directories(firefly_kernels PRIVATE "${FIREFLY_FLASHINFER_INCLUDE_DIR}")
endif()
