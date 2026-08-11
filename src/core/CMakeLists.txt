firefly_add_object_module(firefly_core
    logging.cc
    tensor.cc
)
target_link_libraries(firefly_core PUBLIC mdspan CUDA::cudart)
