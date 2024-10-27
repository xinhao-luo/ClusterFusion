import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer
import numpy as np

def flashinfer_single_decode_e2e(
    hidden, norm_weight1,norm_weight2, eps, 
    k_cache, v_cache, 
    q_proj, k_proj, v_proj, o_proj, 
    gate_proj, up_proj, down_proj, 
    head_dim, head_num, seq_len,
    kv_layout, pos_encoding_mode
):
    torch.set_printoptions(profile="full") 
    residual = hidden
    hidden = flashinfer.norm.rmsnorm(hidden, norm_weight1)
    q = q_proj(hidden).view(1,head_num, head_dim)
    k_new = k_proj(hidden).view(1, head_num, head_dim)
    v_new = v_proj(hidden).view(head_num, 1,  head_dim)
    # ROPE
    indptr = torch.tensor([0, 1]).to(0)  
    offsets = torch.tensor([seq_len]).to(0) # relative pos
    q_new,k_new=flashinfer.apply_rope(q,k_new,indptr,offsets)
   
    k_new = k_new.reshape((head_num, 1, head_dim))
    k = torch.cat((k_cache[:, :seq_len-1, :], k_new), dim=1) 
    v = torch.cat((v_cache[:, :seq_len-1, :], v_new), dim=1)
    
    q_new = q_new.reshape((head_num, head_dim))
    o = flashinfer.single_decode_with_kv_cache(
        q_new, k, v, kv_layout, pos_encoding_mode, use_tensor_cores=False
    )    

    o = o_proj(o.view(1, head_num * head_dim))
    # o = o + residual
    # o = flashinfer.norm.rmsnorm(o, norm_weight2)
    flashinfer.fused_add_rmsnorm(o, residual, norm_weight2, eps)
    print(o)
    residual= o
    print("============== FFN ==============")
    print(F.silu(gate_proj(o)) * up_proj(o))
    o = down_proj(F.silu(gate_proj(o)) * up_proj(o))
    o = o + residual
    print(o)
    # print(o.shape)


hidden_size = 64
seq_len = 8
num_heads = 2
head_dim = 64
ffn_dim = 64

# hidden_size = 4096
# seq_len = 4096
# num_heads = 32
# head_dim = 256
# ffn_dim = 4096

# ===========================================================================
eps = 1e-5

kv_layout = "HND" # [num_kv_heads, kv_len, head_dim]
# * we use roped k cache, I have no idea what flashinfer do...
# * So I chose to rope on the new key and then cat it with k cache
pos_encoding_mode =  "NONE" 

hidden = np.loadtxt("data/h_input") 
hidden = torch.tensor(hidden, dtype=torch.half).to(0)
hidden = hidden.reshape((1, hidden_size))

norm_weight1 = np.loadtxt("data/h_rms_1") 
norm_weight1 = torch.tensor(norm_weight1, dtype=torch.half).to(0)
norm_weight2 = np.loadtxt("data/h_rms_2") 
norm_weight2 = torch.tensor(norm_weight2, dtype=torch.half).to(0)

q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
weights = np.loadtxt('data/h_w_q')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, hidden_size)
with torch.no_grad():
    q_proj.weight.copy_(weights_tensor)

k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
weights = np.loadtxt('data/h_w_k')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, hidden_size)
with torch.no_grad():
    k_proj.weight.copy_(weights_tensor)

v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
weights = np.loadtxt('data/h_w_v')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, hidden_size)
with torch.no_grad():
    v_proj.weight.copy_(weights_tensor)

o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
weights = np.loadtxt('data/h_w_o')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, num_heads * head_dim)
with torch.no_grad():
    o_proj.weight.copy_(weights_tensor)

gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
weights = np.loadtxt('data/h_ffn_1')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, hidden_size)
with torch.no_grad():
    gate_proj.weight.copy_(weights_tensor)

up_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
weights = np.loadtxt('data/h_ffn_3')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, hidden_size)
with torch.no_grad():
    up_proj.weight.copy_(weights_tensor)

down_proj = nn.Linear(ffn_dim, hidden_size, bias=False).to(0).half()
weights = np.loadtxt('data/h_ffn_2')
weights_tensor = torch.tensor(weights, dtype=torch.half).to(0)
weights_tensor = weights_tensor.view(-1, ffn_dim)
with torch.no_grad():
    down_proj.weight.copy_(weights_tensor)

k_cache = np.loadtxt("data/h_k_cache") 
k_cache = torch.tensor(k_cache, dtype=torch.half).to(0)
k_cache = k_cache.reshape((num_heads, seq_len, head_dim)) # ! replace the last dimension later

v_cache = np.loadtxt("data/h_v_cache") 
v_cache = torch.tensor(v_cache, dtype=torch.half).to(0)
v_cache = v_cache.reshape((num_heads, seq_len, head_dim)) # ! replace the last dimension later

flashinfer_single_decode_e2e(
    hidden, 
    norm_weight1, norm_weight2, 
    eps, 
    k_cache, v_cache, 
    q_proj, k_proj, v_proj, o_proj, 
    gate_proj, up_proj, down_proj, 
    head_dim, num_heads,seq_len, kv_layout, pos_encoding_mode
)