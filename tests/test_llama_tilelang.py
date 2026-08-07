"""Correctness test for the tilelang fused llama decode kernel.

Pure-PyTorch reference (no flashinfer dependency), mirrors the formulas in
ClusterFusion's tests/test_llama.py. Requires an H100 (sm90) GPU with
tilelang installed; safe to import on any machine (kernel import is lazy).
"""

import math

import torch

HIDDEN = 4096
HEAD_NUM = 32
HEAD_DIM = 128
QKV_DIM = 3 * HIDDEN


def reference(input, residual, weight_qkv, weight_o, k_cache, v_cache,
              rms_w, eps, cos, sin):
    h = input.float() + residual.float()
    residual_out = h.clone()
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * rms_w.float()

    qkv = x @ weight_qkv.float().T                       # [1, 3*HIDDEN]
    q, k, v = qkv[0].split(HIDDEN)
    q = q.view(HEAD_NUM, HEAD_DIM)
    k = k.view(HEAD_NUM, HEAD_DIM)
    v = v.view(HEAD_NUM, HEAD_DIM)

    half = HEAD_DIM // 2

    def rope(t):                                         # NEOX rotate-half
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * cos - t2 * sin,
                          t2 * cos + t1 * sin], dim=-1)

    q = rope(q)
    k = rope(k)

    seq = k_cache.shape[0]
    K = torch.cat([k_cache.view(seq, HEAD_NUM, HEAD_DIM).float(), k[None]], dim=0)
    V = torch.cat([v_cache.view(seq, HEAD_NUM, HEAD_DIM).float(), v[None]], dim=0)
    scores = torch.einsum("hd,shd->hs", q, K) / math.sqrt(HEAD_DIM)
    probs = torch.softmax(scores, dim=-1)
    o = torch.einsum("hs,shd->hd", probs, V)             # [HEAD_NUM, HEAD_DIM]

    output = o.reshape(1, HIDDEN) @ weight_o.float().T
    return (output.half(), residual_out.half(),
            k.half().unsqueeze(0), v.half().unsqueeze(0))


def _report(name, got, ref):
    diff = (got.float() - ref.float()).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()
    print(f"  {name:10s} MAE={mae:.3e}  max_err={max_err:.3e}")
    return mae, max_err


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU (sm90)"
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tilelang_llama import llama_decoder_layer_sglang

    torch.manual_seed(42)
    dev = "cuda"
    configs = [
        dict(),                                        # ClusterFusion defaults
        dict(threads=256),                             # more threads per CTA
        dict(cluster_size=2),                          # fewer CTAs per head
        dict(tile_s=128),                              # larger KV tile
    ]
    for cfg in configs:
        print(f"config = {cfg or 'default'}")
        for seqlen in (1, 37, 256, 4096):
            print(f"  seqlen = {seqlen}")
            input = torch.randn(1, HIDDEN, dtype=torch.float16, device=dev)
            residual = torch.randn(1, HIDDEN, dtype=torch.float16, device=dev)
            w_qkv = (torch.randn(QKV_DIM, HIDDEN, dtype=torch.float32, device=dev) * 0.1).half()
            w_o = (torch.randn(HIDDEN, HIDDEN, dtype=torch.float32, device=dev) * 0.1).half()
            k_cache = torch.randn(seqlen, HIDDEN, dtype=torch.float16, device=dev)
            v_cache = torch.randn(seqlen, HIDDEN, dtype=torch.float16, device=dev)
            rms_w = (torch.randn(HIDDEN, dtype=torch.float32, device=dev) * 0.1).half()
            angle = torch.rand(HEAD_DIM // 2, dtype=torch.float32, device=dev) * math.pi
            cos, sin = angle.cos(), angle.sin()
            eps = 1e-5

            out, res, k_new, v_new = llama_decoder_layer_sglang(
                input, residual.clone(), w_qkv, w_o, k_cache, v_cache, rms_w,
                eps, cos, sin, **cfg)
            ref_out, ref_res, ref_k, ref_v = reference(
                input, residual, w_qkv, w_o, k_cache, v_cache, rms_w, eps, cos, sin)

            _, e1 = _report("output", out, ref_out)
            _, e2 = _report("residual", res, ref_res)
            _, e3 = _report("k_new", k_new, ref_k)
            _, e4 = _report("v_new", v_new, ref_v)
            assert e2 < 1e-3 and e3 < 1e-2 and e4 < 1e-2 and e1 < 5e-2, "tolerance exceeded"
    print("PASS")


if __name__ == "__main__":
    main()
