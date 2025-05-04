import torch
import torch.nn as nn
import flashinfer
from distfusion import deepseek_decoder_layer
import time

hidden_size = 2048
num_heads = 16
seqlen = 16384
nope_head_dim = hidden_size // num_heads
pe_head_dim = 64
kv_lora_rank = 512
mla_head_dim = kv_lora_rank + pe_head_dim
page_size = 1
bsz = 1

def initialize_rope_embeddings(HEAD_DIM):
    angles = (torch.rand((1, HEAD_DIM), dtype=torch.float32) * (2 * torch.pi)).to(0)
    h_cos = torch.cos(angles)
    h_sin = torch.sin(angles)
    return h_cos, h_sin

# import from deepseek.py
def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

# import from deepseek.py
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def deepseek_decode(hidden, rms_input_weight, rms_ckv_weight, eps, ckv_cache, weight_q, weight_uk, weight_kv, weight_uv, weight_o, cos, sin):
    residual = torch.zeros(hidden.shape).to(0).half()
    flashinfer.fused_add_rmsnorm(hidden, residual, rms_input_weight, eps)
    q = torch.matmul(hidden, weight_q).view(-1, num_heads, nope_head_dim + pe_head_dim)
    q_nope, q_pe = q.split([nope_head_dim, pe_head_dim], dim=-1)
    q_nope = torch.bmm(q_nope.transpose(0, 1), weight_uk).transpose(0, 1)
    latent_cache = torch.matmul(hidden, weight_kv)
    c = latent_cache[..., :kv_lora_rank]
    residual = torch.zeros(c.shape).to(0).half()
    flashinfer.fused_add_rmsnorm(c, residual, rms_ckv_weight, eps)
    latent_cache[..., :kv_lora_rank] = c
    k_pe = latent_cache[..., kv_lora_rank:]
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
    latent_cache[..., kv_lora_rank:] = k_pe
    ckv_cache[-1, :] = latent_cache
    ckv_cache = ckv_cache.unsqueeze(1)
    c = ckv_cache[:, :, :kv_lora_rank].transpose(0, 1)
    attn_weights = torch.bmm(q_nope, c.transpose(1, 2))
    softmax_scale = 1.0 / ((nope_head_dim + pe_head_dim) ** 0.5)
    attn_weights = attn_weights * softmax_scale
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q_nope.dtype)
    attn_output = torch.bmm(attn_weights, c)
    output = torch.bmm(attn_output.transpose(0, 1), weight_uv).transpose(0, 1)
    output = torch.matmul(output.flatten(1, 2), weight_o)

    return output

# without ' * 0.1', the outputs of tilefusion and python both will be 'nan'
def generate_random_weights(shape):
    return (torch.randn(shape) * 0.1).to(0).half()

def test_deepseek_decode_e2e():
    # Generate random weights
    input_tensor = generate_random_weights((1, hidden_size)).to(0).half()
    weight_q_nope = generate_random_weights((hidden_size, num_heads * nope_head_dim)).to(0).half()
    weight_q_pe = generate_random_weights((hidden_size, num_heads * pe_head_dim)).to(0).half()
    weight_uk = generate_random_weights((nope_head_dim, num_heads * kv_lora_rank)).to(0).half()
    weight_kv_nope = generate_random_weights((hidden_size, kv_lora_rank)).to(0).half()
    weight_k_pe = generate_random_weights((hidden_size, pe_head_dim)).to(0).half()
    weight_uv = generate_random_weights((kv_lora_rank, num_heads * nope_head_dim)).to(0).half()
    weight_o = generate_random_weights((num_heads * nope_head_dim, hidden_size)).to(0).half()

    rms_input_weight = generate_random_weights((1, hidden_size)).to(0).half()
    rms_ckv_weight = generate_random_weights((1, kv_lora_rank)).to(0).half()

    # Generate full kv_cache with shape (seqlen, mla_head_dim)
    ckv_cache = generate_random_weights((seqlen, mla_head_dim)).to(0).half()

    # RoPE with cos and sin
    cos, sin = initialize_rope_embeddings(pe_head_dim)
    # Our kernel
    o = deepseek_decoder_layer(
        input_tensor,          
        weight_q_nope,
        weight_q_pe,
        weight_uk,
        weight_kv_nope,
        weight_k_pe,
        weight_uv,                          
        weight_o,              
        ckv_cache,              
        rms_input_weight,      
        rms_ckv_weight,       
        cos,                   
        sin                    
    )
    print(o.shape, o)

    eps = 1e-6
    rms_input_weight = rms_input_weight.reshape((hidden_size,))
    rms_ckv_weight = rms_ckv_weight.reshape((kv_lora_rank,))

    weight_q_nope = weight_q_nope.reshape(hidden_size, num_heads, nope_head_dim)
    weight_q_pe = weight_q_pe.reshape(hidden_size, num_heads, pe_head_dim)
    weight_q = torch.cat((weight_q_nope, weight_q_pe), dim=-1).reshape(hidden_size, -1)
    weight_uk = weight_uk.reshape(nope_head_dim, num_heads, kv_lora_rank).transpose(0, 1)
    weight_uv = weight_uv.reshape(kv_lora_rank, num_heads, nope_head_dim).transpose(0, 1)
    weight_kv = torch.cat((weight_kv_nope, weight_k_pe), dim=-1)

    o_gt = deepseek_decode(input_tensor, rms_input_weight, rms_ckv_weight, eps, ckv_cache, weight_q, weight_uk, weight_kv, weight_uv, weight_o, cos, sin)
    print(o_gt.shape, o_gt)

    mae = (o - o_gt).abs().mean()
    print("Mean Absolute Error (MAE):", mae.item())

    mse = ((o - o_gt) ** 2).mean()
    print("Mean Squared Error (MSE):", mse.item())

    max_error = (o - o_gt).abs().max()
    print("Max Error:", max_error.item())

if __name__ == "__main__":
    test_deepseek_decode_e2e()