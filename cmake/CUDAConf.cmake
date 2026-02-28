if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native") 
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()