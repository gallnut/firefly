set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures built by Firefly")
endif()

option(FIREFLY_ENABLE_NVTX "Enable NVTX ranges for profiler builds" OFF)
option(FIREFLY_ENABLE_RUNTIME_PROFILING "Enable runtime timing and CUDA event profiling" OFF)

add_library(firefly_compile_options INTERFACE)
target_compile_options(firefly_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>
)
target_compile_definitions(firefly_compile_options INTERFACE CUTLASS_SM100_SUPPORT_DISABLED)
if(FIREFLY_ENABLE_NVTX)
    target_compile_definitions(firefly_compile_options INTERFACE FIREFLY_ENABLE_NVTX)
endif()
if(FIREFLY_ENABLE_RUNTIME_PROFILING)
    target_compile_definitions(firefly_compile_options INTERFACE FIREFLY_ENABLE_RUNTIME_PROFILING)
endif()
