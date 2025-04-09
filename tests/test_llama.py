import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import flashinfer
from tilefusion import llama_decoder_layer

def initialize_rope_embeddings(HEAD_DIM):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate random angles uniformly distributed between 0 and 2*pi
    angles = torch.rand((1, HEAD_DIM), dtype=torch.float32, device=device) * (2 * torch.pi)
    # Compute cosine and sine values from the angles
    h_cos = torch.cos(angles)
    h_sin = torch.sin(angles)
    return h_cos, h_sin

# import from llama.py
def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

# import from llama.py
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def sglang_single_decode(hidden, rms_input_weight, rms_attn_weight, eps, kv_cache, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, cos, sin):
    residual = torch.zeros(hidden.shape).to(0).half()
    flashinfer.fused_add_rmsnorm(hidden, residual, rms_input_weight, eps)
    residual = hidden
    qkv_new = qkv_proj(hidden).view(3, 32, head_dim)
    q = qkv_new[0].view(1, 32, head_dim)
    k_new = qkv_new[1].view(1, 32, head_dim)
    v_new = qkv_new[2].view(1, 32, head_dim)
    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
    q = q.reshape(32, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0) 
    v = torch.cat((kv_cache[1], v_new), dim=0)
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout, "NONE", use_tensor_cores=False
    )
    o = o_proj(o.view(1, 32 * head_dim))
    flashinfer.fused_add_rmsnorm(o, residual, rms_attn_weight, eps)
    o_ffn = F.relu(gate_proj(o)) * up_proj(o)
    o = down_proj(o_ffn)
    return o.detach()

# without ' * 0.1', the outputs of tilefusion and py both will be 'nan'
def generate_random_weights(shape):
    return (torch.randn(shape) * 0.1).to(0).half()

def test_sglang_single_decode_e2e(
    hidden_size,
    seq_len,
    num_heads,
    kv_layout
):
    head_dim = hidden_size // num_heads
    ffn_dim_sglang = 11008  # sglang_single_decode uses 11008
    ffn_dim_fuse = 12288    # single_decode_layer uses 12288
    
    # Generate random weights
    input_tensor = generate_random_weights((1, hidden_size)).to(0).half()
    weight_qkv = generate_random_weights((3 * hidden_size, num_heads * head_dim)).to(0).half()
    weight_o = generate_random_weights((num_heads * head_dim, hidden_size)).to(0).half()
    
    # For sglang_single_decode
    gate_up_proj_weight_sglang = generate_random_weights((2 * hidden_size, ffn_dim_sglang)).to(0).half()
    down_proj_weight_sglang = generate_random_weights((ffn_dim_sglang, hidden_size)).to(0).half()
    
    # For single_decode_layer
    gate_up_proj_weight_fuse = torch.zeros((2 * hidden_size, ffn_dim_fuse), dtype=torch.float16, device="cuda")
    down_proj_weight_fuse = torch.zeros((ffn_dim_fuse, hidden_size), dtype=torch.float16, device="cuda")
    
    # Copy the first 11008 dimensions from sglang_single_decode to single_decode_layer
    gate_up_proj_weight_fuse[:, :ffn_dim_sglang] = gate_up_proj_weight_sglang
    down_proj_weight_fuse[:ffn_dim_sglang, :] = down_proj_weight_sglang
    
    rms_input_weight = generate_random_weights((1, hidden_size)).to(0).half()
    rms_attn_weight = generate_random_weights((1, hidden_size)).to(0).half()

    # Generate full kv_cache with shape (2 * seq_len, num_heads * head_dim)
    kv_cache_full = generate_random_weights((2, seq_len, num_heads * head_dim)).to(0).half()

    # RoPE with cos and sin
    cos, sin = initialize_rope_embeddings(head_dim)
    # Ours kernel
    o = llama_decoder_layer(
        input_tensor,          
        weight_qkv,                          
        weight_o,              
        kv_cache_full[0],
        kv_cache_full[1],           
        gate_up_proj_weight_fuse,      
        down_proj_weight_fuse,      
        rms_input_weight,      
        rms_attn_weight,       
        cos,                   
        sin                    
    )
    print(o.shape, o)

    eps = 1e-6
    rms_input_weight = rms_input_weight.reshape((hidden_size,))
    rms_attn_weight = rms_attn_weight.reshape((hidden_size,))

    # Initialize linear layers with the same weights
    qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    gate_proj = nn.Linear(hidden_size, ffn_dim_sglang, bias=False).to(0).half()
    up_proj = nn.Linear(hidden_size, ffn_dim_sglang, bias=False).to(0).half()
    down_proj = nn.Linear(ffn_dim_sglang, hidden_size, bias=False).to(0).half()
    weight_qkv = weight_qkv.reshape(3, hidden_size, -1).transpose(0, 1).reshape(hidden_size, -1)
    qkv_proj.weight.data = weight_qkv.T.contiguous().view(qkv_proj.weight.data.shape)
    o_proj.weight.data = weight_o.T.contiguous().view(o_proj.weight.data.shape)
    gate_proj.weight.data = gate_up_proj_weight_sglang[:hidden_size, :].T.contiguous().view(gate_proj.weight.data.shape)
    up_proj.weight.data = gate_up_proj_weight_sglang[hidden_size:, :].T.contiguous().view(up_proj.weight.data.shape)
    down_proj.weight.data = down_proj_weight_sglang.T.contiguous().view(down_proj.weight.data.shape)

    # Split kv_cache_full into two parts for kv_cache_sgl initialization
    kv_cache_k = kv_cache_full[0].view(seq_len, num_heads, head_dim)
    kv_cache_v = kv_cache_full[1].view(seq_len, num_heads, head_dim)
    kv_cache_sgl = torch.cat([kv_cache_k[:seq_len-1], kv_cache_v[:seq_len-1]], dim=0).view(2, seq_len-1, num_heads, head_dim)
    
    o_sgl = sglang_single_decode(input_tensor, rms_input_weight, rms_attn_weight, eps, kv_cache_sgl, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, cos, sin)
    print(o_sgl.shape, o_sgl)

if __name__ == "__main__":
    test_sglang_single_decode_e2e(4096, 4096, 32, "NHD")