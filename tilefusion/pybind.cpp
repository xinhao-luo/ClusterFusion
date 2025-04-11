#include <torch/extension.h>

torch::Tensor llama_decoder_layer(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor gate_up_proj_weight,
    torch::Tensor down_proj_weight,
    torch::Tensor rms_input_weight,
    torch::Tensor rms_attn_weight,
    torch::Tensor cos,
    torch::Tensor sin
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer, "");
    m.def("deepseek_decoder_layer", &deepseek_decoder_layer, "");
}