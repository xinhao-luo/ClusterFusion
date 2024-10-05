import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import flashinfer


def compute_ffn_hidden_dim(dim, multiple_of):
    hidden_dim = (8 / 3) * dim
    aligned_hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)
    return aligned_hidden_dim

def flashinfer_single_decode_e2e(hidden, norm_weight, eps, k_cache, v_cache, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode):
    residual = hidden
    hidden = flashinfer.norm.rmsnorm(hidden, norm_weight)
    q = q_proj(hidden).view(32, head_dim)
    k_new = k_proj(hidden).view(1, 32, head_dim)
    v_new = v_proj(hidden).view(1, 32, head_dim)
    k = torch.cat((k_cache, k_new), dim=0) 
    v = torch.cat((v_cache, v_new), dim=0)
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout, pos_encoding_mode, use_tensor_cores=False
    )
    o = o_proj(o.view(1, 32 * head_dim))
    flashinfer.fused_add_rmsnorm(o, residual, norm_weight, eps)
    o = down_proj(F.silu(gate_proj(o)) * up_proj(o))
    o = o + residual
    # print(o)
    # print(o.shape)

def flashinfer_batch_decode_e2e(batch, hidden, wrapper, kv_data, norm_weight, eps, k_cache, v_cache, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode):
    residual = hidden
    hidden = flashinfer.norm.rmsnorm(hidden, norm_weight)
    q = q_proj(hidden).view(batch, 32, head_dim)
    k_new = k_proj(hidden).view(batch, 1, 32, head_dim)
    v_new = v_proj(hidden).view(batch, 1, 32, head_dim)
    k = torch.cat((k_cache, k_new), dim=1) 
    v = torch.cat((v_cache, v_new), dim=1)
    o = wrapper.forward(
            q,
            kv_data,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=0.0,
        )
    o = o_proj(o.view(batch, 32 * head_dim))
    flashinfer.fused_add_rmsnorm(o, residual, norm_weight, eps)
    o = down_proj(F.silu(gate_proj(o)) * up_proj(o))
    o = o + residual
    # print(o)
    # print(o.shape)

def test_flashinfer_single_decode_e2e(
    hidden_size,
    seq_len,
    num_heads,
    kv_layout,
    pos_encoding_mode
):
    head_dim = hidden_size // num_heads
    hidden = torch.randn(1, hidden_size).to(0).half()
    norm_weight = torch.randn(hidden_size).to(0).half()
    eps = 1e-6
    ffn_dim = compute_ffn_hidden_dim(hidden_size, 256)
    q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
    k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
    v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
    up_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
    down_proj = nn.Linear(ffn_dim, hidden_size, bias=False).to(0).half()
    k_cache = torch.randn(seq_len, num_heads, head_dim).to(0).half()
    v_cache = torch.randn(seq_len, num_heads, head_dim).to(0).half()
    
    warmup = 5000
    test = 1000

    for i in range(warmup):
        flashinfer_single_decode_e2e(hidden, norm_weight, eps, k_cache, v_cache, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode)
    
    start = time.perf_counter()
    for i in range(test):
        flashinfer_single_decode_e2e(hidden, norm_weight, eps, k_cache, v_cache, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode)
    end = time.perf_counter()
    elapsed_time = end - start
    print("Latency:", elapsed_time / test * 1e3, "ms")

def test_flashinfer_batch_decode_e2e(
    batch_size,
    hidden_size,
    seq_len,
    page_size,
    num_heads,
    kv_layout,
    pos_encoding_mode
):
    head_dim = hidden_size // num_heads
    hidden = torch.randn(batch_size, hidden_size).to(0).half()
    norm_weight = torch.randn(hidden_size).to(0).half()
    eps = 1e-6
    ffn_dim = compute_ffn_hidden_dim(hidden_size, 256)
    q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
    k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
    v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
    up_proj = nn.Linear(hidden_size, ffn_dim, bias=False).to(0).half()
    down_proj = nn.Linear(ffn_dim, hidden_size, bias=False).to(0).half()

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (torch.randn(total_num_pages, 2, page_size, num_heads, head_dim).to(0) / 10).to(0).half()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (seq_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    k_cache = torch.randn(batch_size, seq_len, num_heads, head_dim).to(0).half()
    v_cache = torch.randn(batch_size, seq_len, num_heads, head_dim).to(0).half()
    
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_heads,
        num_heads,
        head_dim,
        page_size,
        "NONE",
        logits_soft_cap=0.0,
        data_type=torch.half,
        q_data_type=torch.half,
    )

    warmup = 5000
    test = 1000

    for i in range(warmup):
        flashinfer_batch_decode_e2e(batch_size, hidden, wrapper, kv_data, norm_weight, eps, k_cache, v_cache, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode)
    
    start = time.perf_counter()
    for i in range(test):
        flashinfer_batch_decode_e2e(batch_size, hidden, wrapper, kv_data, norm_weight, eps, k_cache, v_cache, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, head_dim, kv_layout, pos_encoding_mode)
        end = time.perf_counter()
    elapsed_time = end - start
    print("Latency:", elapsed_time / test * 1e3, "ms")
    

if __name__ == "__main__":
    # test_flashinfer_single_decode_e2e(8192, 2048, 32, "NHD", "ROPE_LLAMA")
    test_flashinfer_batch_decode_e2e(16, 4096, 2048, 8, 32, "NHD", "ROPE_LLAMA")
