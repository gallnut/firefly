firefly_add_object_module(firefly_model_core
    tokenizer.cc
    registry.cc
    config_loader.cc
)
target_link_libraries(firefly_model_core PUBLIC nlohmann_json::nlohmann_json CUDA::cudart ICU::uc ICU::i18n)

file(GLOB model_directories CONFIGURE_DEPENDS LIST_DIRECTORIES TRUE "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(model_directory IN LISTS model_directories)
    if(IS_DIRECTORY "${model_directory}" AND EXISTS "${model_directory}/CMakeLists.txt")
        get_filename_component(model_name "${model_directory}" NAME)
        add_subdirectory("${model_name}")
    endif()
endforeach()

firefly_collect_model_objects(FIREFLY_MODEL_OBJECTS)
set(model_registry_source "${CMAKE_CURRENT_BINARY_DIR}/model_registry.cc")
firefly_generate_model_registry("${model_registry_source}")
target_sources(firefly_model_core PRIVATE "${model_registry_source}")
set(FIREFLY_MODEL_OBJECTS "${FIREFLY_MODEL_OBJECTS}" PARENT_SCOPE)
