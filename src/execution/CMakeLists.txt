firefly_add_object_module(firefly_execution
    engine.cc
    mixed_prefill_decode.cc
    startup.cc
    kv_cache.cc
    prefill.cc
    decode.cc
    dynamic_decode.cc
    static_decode.cc
    output_processing.cc
    speculative_decode.cc
)
target_link_libraries(firefly_execution PUBLIC CUDA::cudart)
