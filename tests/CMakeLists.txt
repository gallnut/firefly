find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME architecture_boundaries
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_architecture.py)

add_executable(scheduler_prefix_cow_test
    scheduler_prefix_cow_test.cc
)

target_link_libraries(scheduler_prefix_cow_test PRIVATE firefly_runtime gtest_main)

add_test(NAME scheduler_prefix_cow_test COMMAND scheduler_prefix_cow_test)

add_executable(model_registry_test model_registry_test.cc)
target_link_libraries(model_registry_test PRIVATE firefly_runtime gtest_main)
add_test(NAME model_registry_test COMMAND model_registry_test)

add_executable(tokenizer_test tokenizer_test.cc)
target_link_libraries(tokenizer_test PRIVATE firefly_runtime gtest_main)
add_test(NAME tokenizer_test COMMAND tokenizer_test)
set_tests_properties(tokenizer_test PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(device_context_stream_test device_context_stream_test.cu)
target_link_libraries(device_context_stream_test PRIVATE firefly_runtime)
set_target_properties(device_context_stream_test PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
add_test(NAME device_context_stream_test COMMAND device_context_stream_test)

add_executable(qwen35_gated_delta_net_accuracy_test qwen35_gated_delta_net_accuracy_test.cu)
target_link_libraries(qwen35_gated_delta_net_accuracy_test PRIVATE firefly_runtime)
set_target_properties(qwen35_gated_delta_net_accuracy_test PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
add_test(NAME qwen35_gated_delta_net_accuracy_test COMMAND qwen35_gated_delta_net_accuracy_test)

if(FIREFLY_HAS_FLASHINFER)
    add_executable(attention_flashinfer_accuracy_test attention_flashinfer_accuracy_test.cu)
    target_link_libraries(attention_flashinfer_accuracy_test PRIVATE firefly_runtime)
    set_target_properties(attention_flashinfer_accuracy_test PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME attention_flashinfer_accuracy_test COMMAND attention_flashinfer_accuracy_test)

    add_executable(qwen_flashinfer_row_test qwen_flashinfer_row_test.cu)
    target_link_libraries(qwen_flashinfer_row_test PRIVATE firefly_runtime)
    set_target_properties(qwen_flashinfer_row_test PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME qwen_flashinfer_row_test COMMAND qwen_flashinfer_row_test)

endif()
