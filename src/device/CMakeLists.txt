firefly_add_object_module(firefly_device
    allocator_cpu.cc
    allocator_cuda.cu
)
target_link_libraries(firefly_device PUBLIC CUDA::cudart)
