firefly_add_object_module(firefly_model_qwen
    model.cu
    ragged_forward.cu
)
firefly_register_model_module(firefly_model_qwen register_qwen_models)
