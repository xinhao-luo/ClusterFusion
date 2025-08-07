import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer
import torch.cuda.nvtx as nvtx
from clusterfusion import llama_decoder_layer

hidden_size = 4096
num_heads = 32
seqlen = 4096
head_dim = hidden_size // num_heads
ffn_dim_gt = 11008  
ffn_dim_fuse = 12288    

torch.manual_seed(42)
# torch.set_printoptions(precision=4, sci_mode=False)

def initialize_rope_embeddings(HEAD_DIM):
    angles = (torch.rand((1, HEAD_DIM), dtype=torch.float32) * (2 * torch.pi)).to(0)
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

def llama_decode(hidden, rms_input_weight, rms_attn_weight, eps, kv_cache, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, cos, sin):
    residual = torch.zeros(hidden.shape).to(0).half()
    flashinfer.fused_add_rmsnorm(hidden, residual, rms_input_weight, eps)
    residual = hidden
    qkv_new = qkv_proj(hidden).view(3, 32, head_dim)
    q = qkv_new[0].view(1, 32, head_dim)
    k_new = qkv_new[1].view(1, 32, head_dim)
    v_new = qkv_new[2].view(1, 32, head_dim)
    head_id = 0
    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)  # RoPE need debug
    q = q.reshape(32, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0) 
    v = torch.cat((kv_cache[1], v_new), dim=0)
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, "NHD", "NONE", use_tensor_cores=False
    )
    o = o_proj(o.view(1, 32 * head_dim))
    # flashinfer.fused_add_rmsnorm(o, residual, rms_attn_weight, eps)
    # o_ffn = F.relu(gate_proj(o)) * up_proj(o)
    # o = down_proj(o_ffn)
    return o.detach()

# without ' * 0.1', the outputs of tilefusion and python both will be 'nan'
def generate_random_weights(shape):
    return (torch.randn(shape) * 0.2).to(0).half()

def test_llama_decode_e2e():
    # Generate random weights
    input_tensor = generate_random_weights((1, hidden_size)).to(0).half()
    weight_qkv = generate_random_weights((3 * hidden_size, num_heads * head_dim)).to(0).half()
    weight_o = generate_random_weights((num_heads * head_dim, hidden_size)).to(0).half()
    
    # For llama_decode
    gate_up_proj_weight_gt = generate_random_weights((2 * hidden_size, ffn_dim_gt)).to(0).half()
    down_proj_weight_gt = generate_random_weights((ffn_dim_gt, hidden_size)).to(0).half()
    
    # For single_decode_layer
    gate_up_proj_weight_fuse = torch.zeros((2 * hidden_size, ffn_dim_fuse), dtype=torch.float16, device="cuda")
    down_proj_weight_fuse = torch.zeros((ffn_dim_fuse, hidden_size), dtype=torch.float16, device="cuda")
    
    # Copy the first 11008 dimensions from llama_decode to single_decode_layer
    gate_up_proj_weight_fuse[:, :ffn_dim_gt] = gate_up_proj_weight_gt
    down_proj_weight_fuse[:ffn_dim_gt, :] = down_proj_weight_gt
    
    rms_input_weight = generate_random_weights((1, hidden_size)).to(0).half()
    rms_attn_weight = generate_random_weights((1, hidden_size)).to(0).half()

    # Generate full kv_cache with shape (2 * seqlen, num_heads * head_dim)
    kv_cache_full = generate_random_weights((2, seqlen, num_heads * head_dim)).to(0).half()

    # RoPE with cos and sin
    cos, sin = initialize_rope_embeddings(head_dim)
    # Our kernel
    o = []
    test_run = 10000
    for i in range(test_run):
        o.append(llama_decoder_layer(
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
        ))


    # for i in range(5):
        # tmp = llama_decoder_layer(
                # input_tensor,          
                # weight_qkv,                          
                # weight_o,              
                # kv_cache_full[0],
                # kv_cache_full[1],           
                # gate_up_proj_weight_fuse,      
                # down_proj_weight_fuse,      
                # rms_input_weight,      
                # rms_attn_weight,       
                # cos,                   
                # sin                    
            # )
        # if not torch.equal(tmp, o):
            # print(tmp)
            # same = False
# 
    # if same:
        # print("Kernel outputs match.")
    # else:
        # print("Kernel outputs differ.")
        # max_error = (tmp - o).abs().max()
        # print(f"Max error between outputs: {max_error.item()}") 
    eps = 1e-6
    rms_input_weight = rms_input_weight.reshape((hidden_size,))
    rms_attn_weight = rms_attn_weight.reshape((hidden_size,))

    # Initialize linear layers with the same weights
    qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    gate_proj = nn.Linear(hidden_size, ffn_dim_gt, bias=False).to(0).half()
    up_proj = nn.Linear(hidden_size, ffn_dim_gt, bias=False).to(0).half()
    down_proj = nn.Linear(ffn_dim_gt, hidden_size, bias=False).to(0).half()
    weight_qkv = weight_qkv.reshape(3, hidden_size, -1).transpose(0, 1).reshape(hidden_size, -1)
    qkv_proj.weight.data = weight_qkv.T.contiguous().view(qkv_proj.weight.data.shape)
    o_proj.weight.data = weight_o.T.contiguous().view(o_proj.weight.data.shape)
    gate_proj.weight.data = gate_up_proj_weight_gt[:hidden_size, :].T.contiguous().view(gate_proj.weight.data.shape)
    up_proj.weight.data = gate_up_proj_weight_gt[hidden_size:, :].T.contiguous().view(up_proj.weight.data.shape)
    down_proj.weight.data = down_proj_weight_gt.T.contiguous().view(down_proj.weight.data.shape)

    # Split kv_cache_full into two parts for kv_cache_gt initialization
    kv_cache_k = kv_cache_full[0].view(seqlen, num_heads, head_dim)
    kv_cache_v = kv_cache_full[1].view(seqlen, num_heads, head_dim)
    kv_cache_gt = torch.cat([kv_cache_k[:seqlen], kv_cache_v[:seqlen]], dim=0).view(2, seqlen, num_heads, head_dim)
    
    nvtx.range_push("llama_decode")
    o_gt = llama_decode(input_tensor, rms_input_weight, rms_attn_weight, eps, kv_cache_gt, qkv_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, cos, sin)
    nvtx.range_pop()

    max_error_overall = 0
    count_of_large_error = 0
    for i in range(test_run):
        mae = (o[i] - o_gt).abs().mean()
        print("Mean Absolute Error (MAE):", mae.item())

        mse = ((o[i] - o_gt) ** 2).mean()
        print("Mean Squared Error (MSE):", mse.item())

        max_error = (o[i] - o_gt).abs().max()
        print("Max Error:", max_error.item())
        max_error_overall = max(max_error, max_error_overall)
        if (max_error.item() > 0.125):
            count_of_large_error += 1
    
    print("Max Error Overall:", max_error_overall.item())
    print("Count of errors > 0.125:", count_of_large_error)
    print(o[0])
    print(o_gt.shape, o_gt)

if __name__ == "__main__":
    test_llama_decode_e2e()