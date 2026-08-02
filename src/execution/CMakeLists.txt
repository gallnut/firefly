firefly_add_object_module(firefly_execution
    engine.cc
    output_processing.cc
)
target_link_libraries(firefly_execution PUBLIC CUDA::cudart)
