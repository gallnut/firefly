firefly_add_object_module(firefly_kernels
    transformer/embedding.cu
    transformer/rms_norm.cu
    transformer/swiglu.cu
    transformer/rope.cu
    transformer/linear.cu
    transformer/residual.cu
    attention/paged_attention.cu
    attention/flashinfer.cu
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
