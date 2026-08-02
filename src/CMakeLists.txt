add_subdirectory(device)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(scheduler)
add_subdirectory(model)
add_subdirectory(execution)

add_library(firefly_runtime STATIC
    $<TARGET_OBJECTS:firefly_device>
    $<TARGET_OBJECTS:firefly_core>
    $<TARGET_OBJECTS:firefly_kernels>
    $<TARGET_OBJECTS:firefly_scheduler>
    $<TARGET_OBJECTS:firefly_model_core>
    ${FIREFLY_MODEL_OBJECTS}
    $<TARGET_OBJECTS:firefly_execution>
)
target_include_directories(firefly_runtime PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(firefly_runtime
    PUBLIC
        firefly_compile_options
        mdspan
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cublas
        ICU::uc
        ICU::i18n
    PRIVATE
        nvidia::cutlass::cutlass
)
set_target_properties(firefly_runtime PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

if(FIREFLY_BUILD_SERVER)
    add_subdirectory(service)
endif()
