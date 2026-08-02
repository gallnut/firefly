set(proto_file "${PROJECT_SOURCE_DIR}/proto/firefly.proto")
set(proto_output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(proto_sources
    "${proto_output_dir}/firefly.pb.cc"
    "${proto_output_dir}/firefly.grpc.pb.cc"
)
set(proto_headers
    "${proto_output_dir}/firefly.pb.h"
    "${proto_output_dir}/firefly.grpc.pb.h"
)

add_custom_command(
    OUTPUT ${proto_sources} ${proto_headers}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_dir}"
    COMMAND $<TARGET_FILE:protobuf::protoc>
    ARGS
        --grpc_out=${proto_output_dir}
        --cpp_out=${proto_output_dir}
        -I ${PROJECT_SOURCE_DIR}/proto
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        ${proto_file}
    DEPENDS "${proto_file}"
    VERBATIM
)

add_library(firefly_proto STATIC ${proto_sources} ${proto_headers})
target_include_directories(firefly_proto PUBLIC "${proto_output_dir}")
target_link_libraries(firefly_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

add_executable(firefly_server
    "${CMAKE_CURRENT_SOURCE_DIR}/../main.cc"
    grpc_adapter.cc
    server.cc
)
target_include_directories(firefly_server PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(firefly_server
    PRIVATE
        firefly_runtime
        firefly_proto
        gRPC::grpc++
        gRPC::grpc++_reflection
)
set_target_properties(firefly_server PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
