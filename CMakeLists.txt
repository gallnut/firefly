cmake_minimum_required(VERSION 3.35)

project(firefly VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CPM)
include(Compile)
include(Targets)

option(FIREFLY_BUILD_SERVER "Build the gRPC serving binary" ON)
option(FIREFLY_USE_FLASHINFER "Use FlashInfer paged decode attention" ON)
include(CTest)

cmake_policy(PUSH)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
include(Deps)
cmake_policy(POP)

add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
