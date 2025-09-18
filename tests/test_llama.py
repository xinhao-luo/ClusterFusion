import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer
import torch.cuda.nvtx as nvtx
import clusterfusion

hidden_size = 4096
num_heads = 32
seqlen = 4096
head_dim = hidden_size // num_heads
ffn_dim_gt = 11008  

torch.manual_seed(42)

# Enable Debug print
debug = 0
print_head = 1
if debug:
    test_run = 10
else:
    test_run = 10000

def initialize_rope_embeddings(HEAD_DIM):
    angles = (torch.rand((1, HEAD_DIM), dtype=torch.float32) * (2 * torch.pi)).to(0)
    h_cos = torch.cos(angles)
    h_sin = torch.sin(angles)
    return h_cos, h_sin

# import from llama.py
def apply_GPT_J_style_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_every_two(q) * sin)
    k_embed = (k * cos) + (rotate_every_two(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

def apply_neox_style_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

# from llama/model.py
def rotate_every_two(x):
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

# import from llama.py
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def llama_decode(hidden, residual, rms_input_weight, rms_attn_weight, eps, kv_cache, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, cos, sin):
    # DEBUG PRINT
    if debug:
        print("----------------------------- python begin -----------------------------")

    flashinfer.fused_add_rmsnorm(hidden, residual, rms_input_weight, eps)
    residual = hidden
    qkv_new = qkv_proj(hidden).view(3, 32, head_dim)
    q = qkv_new[0].view(1, 32, head_dim)
    k_new = qkv_new[1].view(1, 32, head_dim)
    v_new = qkv_new[2].view(1, 32, head_dim)

    # DEBUG PRINT
    if debug: 
        print("normed ref", hidden[..., 0: 128])
        print("before RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 120: 128]}")
        print(f"k_new, head_id = {print_head}: first 8, last 8")
        print(f"{k_new[0, print_head, 0: 8]}")
        print(f"{k_new[0, print_head, 120: 128]}")

    #q, k_new = apply_GPT_J_style_rotary_pos_emb(q, k_new, cos, sin)
    q, k_new = apply_neox_style_rotary_pos_emb(q, k_new, cos, sin)

    # DEBUG PRINT
    if debug: 
        print("after RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 120: 128]}")
        print(f"k_new, head_id = {print_head}: first 8, last 8")
        print(f"{k_new[0, print_head, 0: 8]}")
        print(f"{k_new[0, print_head, 120: 128]}")

    q = q.reshape(32, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0) 
    v = torch.cat((kv_cache[1], v_new), dim=0)
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, "NHD", "NONE", use_tensor_cores=False
    )
    if debug:
        print("attn output O")
        print(f"o, head_id = {print_head}, o")
        print(f"{o[print_head, 0: 128]}")
    o = o_proj(o.view(1, 32 * head_dim))
    if debug:
        print("final output o")
        print(o[0, 0:8])
        print(o[0, 4088:4096])
    # flashinfer.fused_add_rmsnorm(o, residual, rms_attn_weight, eps)
    # o_ffn = F.relu(gate_proj(o)) * up_proj(o)
    # o = down_proj(o_ffn)
    if debug:
        print("-----------------------------  python end  -----------------------------")
    return o.detach()

# without ' * 0.1', the outputs of tilefusion and python both will be 'nan'
def generate_random_weights(shape):
    return (torch.randn(shape) * 0.1).to(0).half()

def test_llama_decode_e2e():
    print(f"seqlen: {seqlen}")
    # Generate random weights
    input_tensor = generate_random_weights((1, hidden_size)).to(0).half()
    residual = generate_random_weights((1, hidden_size)).to(0).half()
    weight_qkv = generate_random_weights((3 * num_heads * head_dim, hidden_size)).to(0).half()
    weight_o = generate_random_weights((num_heads * head_dim, hidden_size)).to(0).half()
    
    # For llama_decode
    gate_up_proj_weight_gt = generate_random_weights((2 * hidden_size, ffn_dim_gt)).to(0).half()
    down_proj_weight_gt = generate_random_weights((ffn_dim_gt, hidden_size)).to(0).half()
    
    rms_input_weight = generate_random_weights((1, hidden_size)).to(0).half()
    rms_attn_weight = generate_random_weights((1, hidden_size)).to(0).half()

    # Generate full kv_cache with shape (2 * seqlen, num_heads * head_dim)
    kv_cache_full = generate_random_weights((2, seqlen, num_heads * head_dim)).to(0).half()

    # RoPE with cos and sin
    cos, sin = initialize_rope_embeddings(head_dim // 2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    # Our kernel
    o = []
    for i in range(test_run):
        tmp_residual = residual.clone()
        output, _, k, v = clusterfusion.llama_decoder_layer_sglang(
            input_tensor,          
            tmp_residual,
            weight_qkv,                          
            weight_o,              
            kv_cache_full[0],
            kv_cache_full[1],           
            rms_input_weight,      
            1e-6,
            cos,                   
            sin                    
        )
        o.append(output)

    eps = 1e-6
    rms_input_weight = rms_input_weight.reshape((hidden_size,))
    rms_attn_weight = rms_attn_weight.reshape((hidden_size,))

    # Initialize linear layers with the same weights
    qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    gate_proj = nn.Linear(hidden_size, ffn_dim_gt, bias=False).to(0).half()
    up_proj = nn.Linear(hidden_size, ffn_dim_gt, bias=False).to(0).half()
    down_proj = nn.Linear(ffn_dim_gt, hidden_size, bias=False).to(0).half()
    qkv_proj.weight.data = weight_qkv.contiguous().view(qkv_proj.weight.data.shape)
    o_proj.weight.data = weight_o.contiguous().view(o_proj.weight.data.shape)
    gate_proj.weight.data = gate_up_proj_weight_gt[:hidden_size, :].T.contiguous().view(gate_proj.weight.data.shape)
    up_proj.weight.data = gate_up_proj_weight_gt[hidden_size:, :].T.contiguous().view(up_proj.weight.data.shape)
    down_proj.weight.data = down_proj_weight_gt.T.contiguous().view(down_proj.weight.data.shape)

    # Split kv_cache_full into two parts for kv_cache_gt initialization
    kv_cache_k = kv_cache_full[0].view(seqlen, num_heads, head_dim)
    kv_cache_v = kv_cache_full[1].view(seqlen, num_heads, head_dim)
    kv_cache_gt = torch.cat([kv_cache_k[:seqlen], kv_cache_v[:seqlen]], dim=0).view(2, seqlen, num_heads, head_dim)
    
    nvtx.range_push("llama_decode")
    o_gt = llama_decode(input_tensor, residual, rms_input_weight, rms_attn_weight, eps, kv_cache_gt, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, cos, sin)
    nvtx.range_pop()
    print(o_gt.shape)
    print("o_gt.abs.mean():", o_gt.abs().mean().item())
    print("Ours[..., 0: 128]", o[0][..., 0:128])
    print("Ref[..., 0: 128]", o_gt[..., 0:128])
    max_error_list = []
    min_error_list = []
    mse_list = []
    mae_list = []
    for i in range(test_run):
        diff = (o[i] - o_gt).abs()
        mae = diff.mean()
        mae_list.append(mae)

        mse = (diff ** 2).mean()
        mse_list.append(mse)

        max_error = diff.max()
        max_error_list.append(max_error)

        max_error_pos = torch.argmax(diff).item()
        # print(f"Run {i}: Max Error {max_error.item()} at position {max_error_pos}")

    print(f"Max Error in MSE of {test_run} runs", max(mse_list).item())
    print(f"Min Error in MSE of {test_run} runs", min(mse_list).item())
    print(f"Max Error in MAE of {test_run} runs", max(mae_list).item())
    print(f"Min Error in MAE of {test_run} runs", min(mae_list).item())
    print(f"Max Error in Max Errors of {test_run} runs", max(max_error_list).item())
    print(f"Min Error in Max Errors of {test_run} runs", min(max_error_list).item())
    print(f"Count of Max Errors > 0.1: {sum(e.item() > 0.1 for e in max_error_list)}")

    max_error_value = max(max_error_list).item()
    max_error_index = max_error_list.index(max(max_error_list))
    print(f"Max Error occurs at run {max_error_index}, value: {max_error_value}")

if __name__ == "__main__":
    test_llama_decode_e2e()