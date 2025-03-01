#include <torch/extension.h>

torch::Tensor single_decode_layer(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor kv_cache,
    torch::Tensor gate_up_proj_weight,
    torch::Tensor down_proj_weight,
    torch::Tensor rms_input_weight,
    torch::Tensor rms_attn_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("single_decode_layer", &single_decode_layer, "");
}