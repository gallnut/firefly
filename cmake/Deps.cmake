# External development overrides
#
# Leave this empty for normal builds. It exists for testing local FlashInfer changes
# or for fully offline builds with a pre-populated source tree.
set(FIREFLY_FLASHINFER_SOURCE_DIR "" CACHE PATH "Optional local FlashInfer source tree")

# System and toolchain dependencies

# CUDA runtime and cuBLAS provide device execution and GEMM support.
find_package(CUDAToolkit REQUIRED)

# ICU executes the Unicode-aware pre-split regex stored in tokenizer.json.
find_package(ICU REQUIRED COMPONENTS uc i18n)

if(FIREFLY_BUILD_SERVER)
    # gRPC and Protobuf implement Firefly's network serving protocol.
    find_package(gRPC REQUIRED)
    find_package(Protobuf REQUIRED)
endif()

# CPM-managed source dependencies

# mdspan provides the multidimensional TensorView implementation.
CPMAddPackage(
    NAME mdspan
    GITHUB_REPOSITORY kokkos/mdspan
    GIT_TAG stable
    OPTIONS
        "MDSPAN_CXX_STANDARD 20"
        "MDSPAN_ENABLE_TESTS OFF"
        "MDSPAN_ENABLE_BENCHMARKS OFF"
)

# CUTLASS provides replaceable CUDA template primitives used by optimized kernels.
CPMAddPackage(
    NAME cutlass
    GITHUB_REPOSITORY NVIDIA/cutlass
    GIT_TAG v4.3.5
    OPTIONS
        "CUTLASS_ENABLE_HEADERS_ONLY ON"
        "CUTLASS_ENABLE_TESTS OFF"
        "CUTLASS_ENABLE_EXAMPLES OFF"
        "CUTLASS_ENABLE_TOOLS OFF"
        "CUTLASS_UNITY_BUILD ON"
        "CUTLASS_ENABLE_SM100_SUPPORT OFF"
        "CUTLASS_ENABLE_GDC OFF"
)

# nlohmann/json parses model configuration and service payloads.
CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.11.3
)

if(FIREFLY_USE_FLASHINFER)
    # FlashInfer supplies the optional high-performance paged-attention backend.
    if(FIREFLY_FLASHINFER_SOURCE_DIR)
        set(flashinfer_SOURCE_DIR "${FIREFLY_FLASHINFER_SOURCE_DIR}")
    else()
        CPMAddPackage(
            NAME flashinfer
            VERSION 0.6.14
            URL https://github.com/flashinfer-ai/flashinfer/archive/refs/tags/v0.6.14.tar.gz
            DOWNLOAD_ONLY YES
        )
    endif()

    if(EXISTS "${flashinfer_SOURCE_DIR}/include/flashinfer/attention/decode.cuh")
        set(FIREFLY_HAS_FLASHINFER ON)
        set(FIREFLY_FLASHINFER_INCLUDE_DIR "${flashinfer_SOURCE_DIR}/include")
    else()
        message(FATAL_ERROR
            "FlashInfer headers were not found under ${flashinfer_SOURCE_DIR}. "
            "Disable FIREFLY_USE_FLASHINFER or provide FIREFLY_FLASHINFER_SOURCE_DIR."
        )
    endif()
else()
    set(FIREFLY_HAS_FLASHINFER OFF)
endif()

if(BUILD_TESTING)
    # GoogleTest is used only by Firefly's C++ unit tests.
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.17.0
        OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
    )
endif()
