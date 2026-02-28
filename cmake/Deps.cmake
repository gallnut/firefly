# kokkos/mdspan
CPMAddPackage(
  NAME mdspan
  GITHUB_REPOSITORY kokkos/mdspan
  GIT_TAG stable
  OPTIONS
    "MDSPAN_CXX_STANDARD 20"
    "MDSPAN_ENABLE_TESTS OFF"
    "MDSPAN_ENABLE_BENCHMARKS OFF"
)

# nvidia/cutlass
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
    "CUTLASS_ENABLE_SM100_SUPPORT OFF"   # Disable bleeding edge
    "CUTLASS_ENABLE_GDC OFF"             # Disable Grid Dependency Control
    "CUTLASS_NVCC_ARCHS_ENABLED 89"      # Try to restrict archs
    "CUTLASS_NVCC_ARCHS_SUPPORTED 89"    # Try to restrict archs
)

# nlohmann/json
CPMAddPackage(
  NAME nlohmann_json
  GITHUB_REPOSITORY nlohmann/json
  VERSION 3.11.3
)

# google/googletest
CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v1.17.0
    VERSION 1.17.0
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
)

# grpc
find_package(gRPC REQUIRED)

# protobuf
find_package(Protobuf REQUIRED)
