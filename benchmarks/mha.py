import torch
import flash_attn_interface
import itertools

import torch.utils.benchmark as benchmark
import flashinfer

def timeit(fn, *args, **kwargs):
    torch.cuda.synchronize()

    # Warmup
    for _ in range(0):
        fn(*args, **kwargs)
    
    # Benchmark using PyTorch Timer
    t = benchmark.Timer(
        stmt='fn(*args, **kwargs)',
        globals={'fn': fn, 'args': args, 'kwargs': kwargs}
    )
    
    # Measure execution time
    measurement = t.timeit(1)  # Runs the function 20 times
    avg_time = measurement.mean  # Average time in seconds

    return avg_time


def test_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    q_dtype,
    kv_dtype,
):
    q = torch.randn(batch_size, num_qo_heads, head_dim).to(0).to(q_dtype)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, num_kv_heads, page_size, head_dim).to(0)
        if kv_layout == "HND"
        else torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim).to(0)
    ).to(kv_dtype)
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq     
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(            
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )

    time = timeit(wrapper.run, q, kv_data)* 1000. * 1000.
    return time

def main():
    # causal = True
    causal = False
    dtype=torch.float16

    torch.manual_seed(42)  

    model_configs = [
        ("Llama-2-7B", 32, 32, 128)             # model_name, nheads_q, nheads_kv, headdim 
    ]

    all_batch_configs = []

    all_batch_configs.extend(itertools.product(
        # [1024, 4096, 8192, 16384],  # context_seqlen
        [16384],  # context_seqlen
        [16],  # num_requests  batch_size
        [1],  # query_seqlen
    ))

    num_caches = max(reqs for _, reqs, _ in all_batch_configs)      # max_num_seqs = 16.  batch_size
    cache_seqlen = max(seqlen for seqlen, _, _ in all_batch_configs)        # max_context_seqlen = 65536    

    for model_name, nheads_q, nheads_kv, headdim in model_configs:    # model config
        k_cache = torch.randn(
            (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
        )
        v_cache = torch.randn(
            (num_caches, cache_seqlen, nheads_kv, headdim), device="cuda", dtype=dtype
        )
        print(f"***{model_name}***")
        print(f"QHEADS:{nheads_q}, KVHEADS:{nheads_kv}, HEADDIM:{headdim}")

        for context_seqlen, num_requests, query_seqlen in all_batch_configs:
            print(f"\n### {num_requests = }, {context_seqlen = }, {query_seqlen = }, {causal = } ###")
            q = torch.randn((num_requests, query_seqlen, nheads_q, headdim), device="cuda", dtype=dtype)
            cache_idxs = torch.randperm(num_caches, dtype=torch.int32, device="cuda")[:num_requests]
            cache_seqlens = torch.tensor(
                [context_seqlen] * num_requests, dtype=torch.int32, device="cuda"
            )
            
            # fa3_time_one_split = timeit(            
            #     flash_attn_interface.flash_attn_with_kvcache,
            #     q=q,
            #     k_cache=k_cache,
            #     v_cache=v_cache,
            #     cache_seqlens=cache_seqlens,
            #     cache_batch_idx=cache_idxs,
            #     causal=causal,
            #     gqa_parallel=False,
            #     num_splits=1,
            # ) * 1000. * 1000.

            fa3_time_heuristic = timeit(        
                flash_attn_interface.flash_attn_with_kvcache,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_idxs,
                causal=causal,
                gqa_parallel=False,
                num_splits=0,
                max_seqlen_k_hint=context_seqlen
            ) * 1000. * 1000.

            # decode
            flashinfer_time = test_batch_decode_with_paged_kv_cache(num_requests,context_seqlen,16,nheads_kv,nheads_q,headdim,"NHD",dtype,dtype)

            # print(f'fa3_one_split: {fa3_time_one_split:.2f}us')
            print(f'fa3_heuristic: {fa3_time_heuristic:.2f}us')
            print(f'flashinfer: {flashinfer_time:.2f}us')

if __name__ == "__main__":
    main()