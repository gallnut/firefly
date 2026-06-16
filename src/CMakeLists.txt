add_library(firefly_core OBJECT
    allocator_cpu.cc
    allocator_cuda.cu
    tensor.cc
    kernels/embedding.cu
    kernels/norm.cu
    kernels/activation.cu
    kernels/rope.cu
    kernels/sampling.cu
    kernels/linear.cu
    kernels/attention.cu
    kernels/attention_torch.cu
    kernels/add.cu
    tokenizer.cc
    scheduler.cc
    engine.cc
    model_registry.cc
    radix_tree.cc
    model/qwen_model.cu
)

target_include_directories(firefly_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(firefly_core PUBLIC mdspan nlohmann_json::nlohmann_json CUDA::cublas)
target_link_libraries(firefly_core PRIVATE nvidia::cutlass::cutlass)
if(FIREFLY_USE_TORCH_SDPA AND Torch_FOUND)
    target_compile_definitions(firefly_core PRIVATE FIREFLY_USE_TORCH_SDPA)
    target_include_directories(firefly_core PRIVATE
        ${FIREFLY_TORCH_ROOT}/include
        ${FIREFLY_TORCH_ROOT}/include/torch/csrc/api/include
    )
    target_link_libraries(firefly_core PRIVATE
        ${FIREFLY_TORCH_LIB_DIR}/libtorch_cuda.so
        ${FIREFLY_TORCH_LIB_DIR}/libtorch_cpu.so
        ${FIREFLY_TORCH_LIB_DIR}/libtorch.so
        ${FIREFLY_TORCH_LIB_DIR}/libc10_cuda.so
        ${FIREFLY_TORCH_LIB_DIR}/libc10.so
    )
endif()
set_target_properties(firefly_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

set(PROTO_SRC_DIR ${CMAKE_SOURCE_DIR}/proto)
set(PROTO_BUILD_DIR ${CMAKE_BINARY_DIR}/grpc_out)
file(MAKE_DIRECTORY ${PROTO_BUILD_DIR})

set(PROTO_FILE ${PROTO_SRC_DIR}/firefly.proto)
set(PROTO_SRCS ${PROTO_BUILD_DIR}/firefly.pb.cc ${PROTO_BUILD_DIR}/firefly.grpc.pb.cc)
set(PROTO_HDRS ${PROTO_BUILD_DIR}/firefly.pb.h ${PROTO_BUILD_DIR}/firefly.grpc.pb.h)

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND $<TARGET_FILE:protobuf::protoc>
    ARGS --grpc_out=${PROTO_BUILD_DIR}
         --cpp_out=${PROTO_BUILD_DIR}
         -I ${PROTO_SRC_DIR}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
)

add_library(firefly_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(firefly_proto PUBLIC ${PROTO_BUILD_DIR} ${PROTO_SRC_DIR})
target_link_libraries(firefly_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

add_executable(firefly_server server.cu grpc_adapter.cc)
target_include_directories(firefly_server PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${PROTO_BUILD_DIR}
)
target_link_libraries(firefly_server PRIVATE firefly_core firefly_proto gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)
if(FIREFLY_USE_TORCH_SDPA AND Torch_FOUND)
    set_target_properties(firefly_server PROPERTIES BUILD_RPATH "${FIREFLY_TORCH_LIB_DIR}")
endif()
set_target_properties(firefly_server PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
