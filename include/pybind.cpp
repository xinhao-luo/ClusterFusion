#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sm90(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm90(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
);

void llama_decoder_layer_batch_sglang_sm90(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
);

torch::Tensor deepseek_decoder_layer(
    torch::Tensor input,
    torch::Tensor weight_q_nope,
    torch::Tensor weight_q_pe,
    torch::Tensor weight_uk,
    torch::Tensor weight_kv_nope,
    torch::Tensor weight_k_pe,
    torch::Tensor weight_uv,
    torch::Tensor weight_o,
    torch::Tensor ckv_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor rms_ckv_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

torch::Tensor rmsnorm(
    torch::Tensor input,
    torch::Tensor weight
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm120(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
);

void llama_decoder_layer_batch_sglang_sm120(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
);

#ifdef COMPILE_SM90
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer_sm90, "");
    m.def("llama_decoder_layer_sglang", &llama_decoder_layer_sglang_sm90, "");
    m.def("llama_decoder_layer_batch_decode_sglang", &llama_decoder_layer_batch_sglang_sm90, "");
    m.def("deepseek_decoder_layer", &deepseek_decoder_layer, "");
    m.def("rmsnorm", &rmsnorm, "");
}
#endif
#ifdef COMPILE_SM120
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer_sm120, "");
    m.def("llama_decoder_layer_sglang", &llama_decoder_layer_sglang_sm120, "");
    m.def("llama_decoder_layer_batch_decode_sglang", &llama_decoder_layer_batch_sglang_sm120, "");
}
#endif