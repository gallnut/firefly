cmake_minimum_required(VERSION 3.35)

project(firefly VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CPM)
include(CXXConf)
include(CUDAConf)
include(Deps)

option(FIREFLY_USE_TORCH_SDPA "Use PyTorch CUDA flash attention for external SDPA backend" ON)
if(FIREFLY_USE_TORCH_SDPA)
    set(FIREFLY_TORCH_ROOT "${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/torch")
    set(FIREFLY_TORCH_LIB_DIR "${FIREFLY_TORCH_ROOT}/lib")
    if(EXISTS "${FIREFLY_TORCH_LIB_DIR}/libtorch_cuda.so")
        message(STATUS "Firefly Torch SDPA backend enabled")
        set(Torch_FOUND TRUE)
    else()
        message(WARNING "Firefly Torch SDPA backend requested but venv Torch CUDA libraries were not found; disabling")
        set(FIREFLY_USE_TORCH_SDPA OFF)
    endif()
endif()

# Disable bleeding edge SM100 support which breaks on non-Blackwell
string(REPLACE "-DCUTLASS_SM100_FAMILY_ARCHS_ENABLED" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-UCUTLASS_SM100_FAMILY_ARCHS_ENABLED>
    $<$<COMPILE_LANGUAGE:CUDA>:-UCUTLASS_SM100_SUPPORT_ENABLED>
)
add_compile_definitions(CUTLASS_SM100_SUPPORT_DISABLED)

add_subdirectory(src)

# option(BUILD_TESTS "Build tests" ON)

# if (BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(test)
# endif()
