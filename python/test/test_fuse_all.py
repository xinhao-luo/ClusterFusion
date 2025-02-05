import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import flashinfer
from fuse_all import single_decode_layer

def sglang_single_decode(hidden, norm_weight, eps, kv_cache, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode):
    hidden = flashinfer.norm.rmsnorm(hidden, norm_weight)
    residual = hidden
    qkv_new = qkv_proj(hidden).view(3, 32, head_dim)
    q = qkv_new[0].view(32, head_dim)
    k_new = qkv_new[1].view(1, 32, head_dim)
    v_new = qkv_new[2].view(1, 32, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0) 
    v = torch.cat((kv_cache[1], v_new), dim=0)
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout, pos_encoding_mode, use_tensor_cores=False
    )
    o = o_proj(o.view(1, 32 * head_dim))
    flashinfer.fused_add_rmsnorm(o, residual, norm_weight, eps)
    o = down_proj(F.relu(gate_proj(o)) * up_proj(o))
    return o

def generate_random_weights(shape):
    return (torch.rand(shape) * 0.001 + 0.003).to(0).half()

def test_flashinfer_single_decode_e2e(
    hidden_size,
    seq_len,
    num_heads,
    kv_layout,
    pos_encoding_mode
):
    head_dim = hidden_size // num_heads
    ffn_dim = 4096
    
    # Generate random weights in the range [0.005, 0.01]
    hidden = torch.randn(1, hidden_size).to(0).half()
    input_tensor = generate_random_weights((1, hidden_size))
    weight_qkv = generate_random_weights((3 * hidden_size, num_heads * head_dim))
    weight_o = generate_random_weights((num_heads * head_dim, hidden_size))
    
    # Generate full kv_cache with shape (2 * seq_len, num_heads * head_dim)
    kv_cache_full = generate_random_weights((2 * seq_len, num_heads * head_dim))

    gate_up_proj_weight = generate_random_weights((2 * hidden_size, ffn_dim))
    down_proj_weight = generate_random_weights((ffn_dim, hidden_size))
    rms_input_weight = generate_random_weights((1, hidden_size))
    rms_attn_weight = generate_random_weights((1, hidden_size))
    cos = torch.full((1, head_dim), 1.0, dtype=torch.float32).to(0)
    sin = torch.full((1, head_dim), 0.0, dtype=torch.float32,).to(0)  
    
    # Use the same weights for both computations
    o = single_decode_layer(
        input_tensor,          
        weight_qkv,                          
        weight_o,              
        kv_cache_full,                            
        gate_up_proj_weight,      
        down_proj_weight,      
        rms_input_weight,      
        rms_attn_weight,       
        cos,                   
        sin                    
    )
    print(o.shape, o)

    eps = 1e-5
    norm_weight = generate_random_weights((hidden_size,))
    
    # Initialize linear layers with the same weights
    qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
    up_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
    down_proj = nn.Linear(ffn_dim, hidden_size, bias=False).to(0).half()
    qkv_proj.weight.data = weight_qkv.view(qkv_proj.weight.data.shape)
    o_proj.weight.data = weight_o.view(o_proj.weight.data.shape)
    gate_proj.weight.data = gate_up_proj_weight[:hidden_size, :].view(gate_proj.weight.data.shape)
    up_proj.weight.data = gate_up_proj_weight[hidden_size:, :].view(up_proj.weight.data.shape)
    down_proj.weight.data = down_proj_weight.view(down_proj.weight.data.shape)

    # Split kv_cache_full into two parts for kv_cache_sgl initialization
    kv_cache_k = kv_cache_full[:seq_len].view(seq_len, num_heads, head_dim)
    kv_cache_v = kv_cache_full[seq_len:2*seq_len].view(seq_len, num_heads, head_dim)
    # Initialize kv_cache for single_decode_layer
    kv_cache_sgl = torch.cat([kv_cache_k[:seq_len-1], kv_cache_v[:seq_len-1]], dim=0).view(2, seq_len-1, num_heads, head_dim)
    
    o_sgl = sglang_single_decode(hidden, norm_weight, eps, kv_cache_sgl, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode)
    print(o_sgl.shape, o_sgl)
    
    max_diff = torch.max(torch.abs(o - o_sgl)).item()
    print(f"Maximum difference: {max_diff}")
    assert max_diff < 5 * 1e-4, "The maximum difference is too large!"

if __name__ == "__main__":
    test_flashinfer_single_decode_e2e(4096, 4096, 32, "NHD", "NONE")