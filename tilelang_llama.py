"""TileLang rewrite of ClusterFusion's fused Llama decode kernel (H100, sm90).

Mirrors include/H100/llama/kernel_sglang.cuh from xinhao-luo/ClusterFusion:
fused add + RMSNorm -> QKV projection -> cluster all-reduce -> NEOX RoPE ->
flash-decoding attention with KV split across the cluster -> two-level softmax
merge -> O projection with global atomicAdd.

One cluster (cluster_size CTAs) owns one attention head. Cluster collectives
are log2(cluster_size)-hop ring all-reduces over DSM: T.copy_cluster lowers to
cp.async.bulk.shared::cluster + remote mbarrier expect_tx, and each hop sends
the *accumulated* partial result to (rank + 2^i) % cluster_size, doubling the
covered range per hop (same scheme as dsm.cuh, generalized beyond size 4).
A cluster_sync after each hop prevents a fast CTA from overwriting a peer's
receive buffer before the peer consumed it.

Parallelism knobs (threads, cluster_size, tile sizes, pipeline stages) are
parameters of _build_kernel; compiled kernels are cached per config.

Untested: this machine has no GPU. Compile and validate on an H100 with
tests/test_llama_tilelang.py before use.
"""

import torch
import tilelang
import tilelang.language as T

# Llama-2-7B model dims (mirrors include/H100/llama/config.h)
HIDDEN = 4096
HEAD_NUM = 32
HEAD_DIM = 128
QKV_DIM = 3 * HIDDEN

LOG2E = 1.4426950408889634
QK_SCALE = (1.0 / (HEAD_DIM ** 0.5)) * LOG2E  # exp2-domain softmax scale

_MAX_SMEM = 227 * 1024  # sm90 max dynamic shared memory per CTA


def _ring_allreduce_vec(rank, send, recv, cbar, bar0, op, n, hops, cluster_size):
    # Ring all-reduce on a half vector; `send` holds the accumulated partial
    # and ends up holding the full reduction on every CTA.
    for i, hop in enumerate(hops):
        T.sync_threads()
        T.copy_cluster(send, recv, dst_block=(rank + hop) % cluster_size,
                       remote_barrier=cbar[bar0 + i])
        T.mbarrier_wait_parity(cbar[bar0 + i], 0)
        if op == "sum":
            for j in T.Parallel(n):
                send[j] = send[j] + recv[j]
        else:
            for j in T.Parallel(n):
                send[j] = T.max(send[j], recv[j])
        T.sync_threads()
        T.cluster_sync()


def _ring_allreduce_scalar(rank, send, recv, cbar, bar0, op, hops, cluster_size):
    # Same protocol on one float (packed in a 16B buffer to keep the
    # cp.async.bulk fast path).
    for i, hop in enumerate(hops):
        T.sync_threads()
        T.copy_cluster(send, recv, dst_block=(rank + hop) % cluster_size,
                       remote_barrier=cbar[bar0 + i])
        T.mbarrier_wait_parity(cbar[bar0 + i], 0)
        if T.get_thread_binding() == 0:
            if op == "sum":
                send[0] = send[0] + recv[0]
            else:
                send[0] = T.max(send[0], recv[0])
        T.sync_threads()
        T.cluster_sync()


def _estimate_smem_bytes(threads, cluster_size, tile_in, tile_s, tile_o, num_stages):
    dim_per_block = HIDDEN // cluster_size
    n_hops = max(0, cluster_size.bit_length() - 1)
    total = (
        dim_per_block * 4                              # s_x (f32)
        + dim_per_block * 2                            # s_xn (half)
        + HEAD_DIM * tile_in * 2 * num_stages          # s_w (qkv weight tiles)
        + 3 * HEAD_DIM * 2 * 2                         # s_qkv + s_qkv_recv
        + tile_s * HEAD_DIM * 2 * 2 * num_stages       # s_k + s_v
        + tile_s * 4                                   # s_p
        + tile_o * HEAD_DIM * 2 * num_stages           # s_wo
        + HEAD_DIM * 2 * 2                             # s_o_send + s_o_recv
        + 64                                           # scalar shared buffers
        + 8 * 5 * n_hops                               # cluster barriers
    )
    return total


def _build_kernel(threads=128, cluster_size=4, tile_in=64, tile_s=64,
                  tile_o=64, num_stages=2):
    if cluster_size < 1 or cluster_size & (cluster_size - 1):
        raise ValueError(f"cluster_size must be a power of two, got {cluster_size}")
    if HIDDEN % cluster_size:
        raise ValueError(f"HIDDEN={HIDDEN} not divisible by cluster_size={cluster_size}")
    dim_per_block = HIDDEN // cluster_size
    if dim_per_block % tile_in or dim_per_block % tile_o:
        raise ValueError("dim_per_block must be divisible by tile_in and tile_o")
    smem = _estimate_smem_bytes(threads, cluster_size, tile_in, tile_s, tile_o, num_stages)
    if smem > _MAX_SMEM:
        raise ValueError(f"estimated shared memory {smem} B exceeds {_MAX_SMEM} B; "
                         "reduce tile sizes or num_stages")

    hops = [1 << i for i in range(cluster_size.bit_length() - 1)]  # 1,2,...,cs/2
    n_hops = len(hops)
    seqlen = T.dynamic("seqlen")

    @T.prim_func
    def kernel(
        Input: T.Tensor((1, HIDDEN), "float16"),
        Residual: T.Tensor((1, HIDDEN), "float16"),   # updated in place
        Wqkv: T.Tensor((QKV_DIM, HIDDEN), "float16"),  # torch Linear layout [out, in]
        Wo: T.Tensor((HIDDEN, HIDDEN), "float16"),
        KCache: T.Tensor((seqlen, HIDDEN), "float16"),  # [seq, n_heads*head_dim] NHD
        VCache: T.Tensor((seqlen, HIDDEN), "float16"),
        RmsW: T.Tensor((HIDDEN,), "float16"),
        Cos: T.Tensor((HEAD_DIM // 2,), "float32"),
        Sin: T.Tensor((HEAD_DIM // 2,), "float32"),
        Output: T.Tensor((1, HIDDEN), "float16"),      # pre-zeroed; atomicAdd target
        KNew: T.Tensor((HEAD_NUM, HEAD_DIM), "float16"),
        VNew: T.Tensor((HEAD_NUM, HEAD_DIM), "float16"),
        eps: T.float32,
    ):
        with T.ClusterKernel(HEAD_NUM * cluster_size, threads=threads,
                             cluster_dims=(cluster_size, 1, 1)) as bx:
            head = bx // cluster_size        # one cluster per attention head
            rank = bx % cluster_size         # CTA rank inside the cluster
            tid = T.get_thread_binding()
            neg_inf = -T.infinity("float32")
            kv_per_block = T.ceildiv(T.ceildiv(seqlen, cluster_size), tile_s) * tile_s

            s_x = T.alloc_shared((dim_per_block,), "float32")
            s_xn = T.alloc_shared((dim_per_block,), "float16")
            s_w = T.alloc_shared((HEAD_DIM, tile_in), "float16")
            s_qkv = T.alloc_shared((3 * HEAD_DIM,), "float16")
            s_qkv_recv = T.alloc_shared((3 * HEAD_DIM,), "float16")
            s_k = T.alloc_shared((tile_s, HEAD_DIM), "float16")
            s_v = T.alloc_shared((tile_s, HEAD_DIM), "float16")
            s_p = T.alloc_shared((1, tile_s), "float32")
            s_wo = T.alloc_shared((tile_o, HEAD_DIM), "float16")
            s_o_send = T.alloc_shared((HEAD_DIM,), "float16")
            s_o_recv = T.alloc_shared((HEAD_DIM,), "float16")
            s_ssend = T.alloc_shared((4,), "float32")
            s_srecv = T.alloc_shared((4,), "float32")
            s_ssq = T.alloc_shared((1,), "float32")
            s_rms = T.alloc_shared((1,), "float32")
            s_m = T.alloc_shared((1,), "float32")
            s_l = T.alloc_shared((1,), "float32")
            s_alpha = T.alloc_shared((1,), "float32")
            s_tm = T.alloc_shared((1,), "float32")
            s_tl = T.alloc_shared((1,), "float32")
            s_ct = T.alloc_shared((1,), "float32")
            s_pct = T.alloc_shared((1,), "float32")
            # 5 collectives x n_hops barriers each: RMS, QKV, softmax max/sum, ATTN
            # (max(1, ...) keeps a dummy barrier for cluster_size == 1)
            cbar = T.alloc_cluster_barrier([1] * max(1, 5 * n_hops))

            f_sq = T.alloc_fragment((1, dim_per_block), "float32")
            f_acc = T.alloc_fragment((HEAD_DIM,), "float32")
            f_scores = T.alloc_fragment((1, tile_s), "float32")
            f_p = T.alloc_fragment((1, tile_s), "float32")
            f_o = T.alloc_fragment((HEAD_DIM,), "float32")
            f_pv = T.alloc_fragment((HEAD_DIM,), "float32")
            f_ct = T.alloc_fragment((1, HEAD_DIM), "float32")
            f_oa = T.alloc_fragment((tile_o,), "float32")

            # ---- Stage 1: fused residual add + RMSNorm (cluster-reduced sum of squares)
            for i in T.Parallel(dim_per_block):
                h = T.Cast("float32", Input[0, rank * dim_per_block + i]) \
                    + T.Cast("float32", Residual[0, rank * dim_per_block + i])
                s_x[i] = h
                Residual[0, rank * dim_per_block + i] = T.Cast("float16", h)
                f_sq[0, i] = h * h
            T.reduce_sum(f_sq, s_ssq, dim=1)
            T.sync_threads()
            if tid == 0:
                s_ssend[0] = s_ssq[0]
            _ring_allreduce_scalar(rank, s_ssend, s_srecv, cbar, 0, "sum", hops, cluster_size)
            if tid == 0:
                s_rms[0] = T.rsqrt(s_ssend[0] / HIDDEN + eps)
            T.sync_threads()
            for i in T.Parallel(dim_per_block):
                s_xn[i] = T.Cast("float16", s_x[i] * s_rms[0]
                                 * T.Cast("float32", RmsW[rank * dim_per_block + i]))

            # ---- Stage 2: QKV projection (3 GEMVs over the local hidden slice)
            for j in range(3):
                T.clear(f_acc)
                for t in T.Pipelined(dim_per_block // tile_in, num_stages=num_stages):
                    T.copy(Wqkv[j * HIDDEN + head * HEAD_DIM,
                                rank * dim_per_block + t * tile_in], s_w)
                    for i in T.Parallel(HEAD_DIM):
                        for k in T.serial(tile_in):
                            f_acc[i] += T.Cast("float32", s_w[i, k]) \
                                * T.Cast("float32", s_xn[t * tile_in + k])
                for i in T.Parallel(HEAD_DIM):
                    s_qkv[j * HEAD_DIM + i] = T.Cast("float16", f_acc[i])
            T.sync_threads()
            # ClusterReduce #1: complete q|k|v on every CTA
            _ring_allreduce_vec(rank, s_qkv, s_qkv_recv, cbar, n_hops, "sum",
                                3 * HEAD_DIM, hops, cluster_size)

            # ---- Stage 3: NEOX RoPE on q,k; rank 0 exports the new k/v
            for i in T.Parallel(HEAD_DIM // 2):
                c = Cos[i]
                s = Sin[i]
                q0 = T.Cast("float32", s_qkv[i])
                q1 = T.Cast("float32", s_qkv[i + HEAD_DIM // 2])
                s_qkv[i] = T.Cast("float16", q0 * c - q1 * s)
                s_qkv[i + HEAD_DIM // 2] = T.Cast("float16", q1 * c + q0 * s)
                k0 = T.Cast("float32", s_qkv[HEAD_DIM + i])
                k1 = T.Cast("float32", s_qkv[HEAD_DIM + i + HEAD_DIM // 2])
                s_qkv[HEAD_DIM + i] = T.Cast("float16", k0 * c - k1 * s)
                s_qkv[HEAD_DIM + i + HEAD_DIM // 2] = T.Cast("float16", k1 * c + k0 * s)
            T.sync_threads()
            if rank == 0:
                T.copy(s_qkv[HEAD_DIM:2 * HEAD_DIM], KNew[head, 0])
                T.copy(s_qkv[2 * HEAD_DIM:3 * HEAD_DIM], VNew[head, 0])

            # ---- Stage 4: flash-decoding attention over this CTA's KV segment
            T.clear(f_o)
            T.fill(s_m, neg_inf)
            T.fill(s_l, 0.0)
            for t in T.Pipelined(T.ceildiv(kv_per_block, tile_s), num_stages=num_stages):
                T.copy(KCache[rank * kv_per_block + t * tile_s, head * HEAD_DIM], s_k)
                T.copy(VCache[rank * kv_per_block + t * tile_s, head * HEAD_DIM], s_v)
                for r in T.Parallel(tile_s):
                    f_scores[0, r] = 0.0
                    for d in T.serial(HEAD_DIM):
                        f_scores[0, r] += T.Cast("float32", s_qkv[d]) \
                            * T.Cast("float32", s_k[r, d])
                    f_scores[0, r] = T.if_then_else(
                        rank * kv_per_block + t * tile_s + r < seqlen,
                        f_scores[0, r] * QK_SCALE, neg_inf)
                T.reduce_max(f_scores, s_tm, dim=1)
                T.sync_threads()
                if tid == 0:
                    # s_m == m_new also covers the all-(-inf) tile (alpha=1, o stays 0)
                    m_new = T.max(s_m[0], s_tm[0])
                    s_alpha[0] = T.if_then_else(s_m[0] == m_new, 1.0,
                                                T.exp2(s_m[0] - m_new))
                    s_m[0] = m_new
                T.sync_threads()
                for r in T.Parallel(tile_s):
                    f_p[0, r] = T.if_then_else(f_scores[0, r] == neg_inf, 0.0,
                                               T.exp2(f_scores[0, r] - s_m[0]))
                T.reduce_sum(f_p, s_tl, dim=1)
                T.copy(f_p, s_p)
                T.sync_threads()
                if tid == 0:
                    s_l[0] = s_l[0] * s_alpha[0] + s_tl[0]
                for d in T.Parallel(HEAD_DIM):
                    f_pv[d] = 0.0
                    for r in T.serial(tile_s):
                        f_pv[d] += s_p[0, r] * T.Cast("float32", s_v[r, d])
                    f_o[d] = f_o[d] * s_alpha[0] + f_pv[d]
                T.sync_threads()

            # current token's k/v join the softmax exactly once (rank 0)
            if rank == 0:
                for d in T.Parallel(HEAD_DIM):
                    f_ct[0, d] = T.Cast("float32", s_qkv[d]) \
                        * T.Cast("float32", s_qkv[HEAD_DIM + d])
                T.reduce_sum(f_ct, s_ct, dim=1)
                T.sync_threads()
                if tid == 0:
                    sc = s_ct[0] * QK_SCALE
                    m_ct = T.max(s_m[0], sc)
                    s_alpha[0] = T.if_then_else(s_m[0] == m_ct, 1.0,
                                                T.exp2(s_m[0] - m_ct))
                    s_pct[0] = T.exp2(sc - m_ct)
                    s_m[0] = m_ct
                    s_l[0] = s_l[0] * s_alpha[0] + s_pct[0]
                T.sync_threads()
                for d in T.Parallel(HEAD_DIM):
                    f_o[d] = f_o[d] * s_alpha[0] \
                        + s_pct[0] * T.Cast("float32", s_qkv[2 * HEAD_DIM + d])

            # ---- Stage 5: cluster-level softmax merge (max, rescale, sum, normalize)
            T.sync_threads()
            if tid == 0:
                s_ssend[0] = s_m[0]
            _ring_allreduce_scalar(rank, s_ssend, s_srecv, cbar, 2 * n_hops, "max",
                                   hops, cluster_size)
            if tid == 0:
                s_alpha[0] = T.exp2(s_m[0] - s_ssend[0])  # 0 if this CTA had no valid row
                s_l[0] = s_l[0] * s_alpha[0]
            T.sync_threads()
            for d in T.Parallel(HEAD_DIM):
                f_o[d] = f_o[d] * s_alpha[0]
            T.sync_threads()
            if tid == 0:
                s_ssend[0] = s_l[0]
            _ring_allreduce_scalar(rank, s_ssend, s_srecv, cbar, 3 * n_hops, "sum",
                                   hops, cluster_size)
            for d in T.Parallel(HEAD_DIM):
                s_o_send[d] = T.Cast("float16", f_o[d] / s_ssend[0])
            T.sync_threads()
            # ClusterReduce #2: full head output on every CTA
            _ring_allreduce_vec(rank, s_o_send, s_o_recv, cbar, 4 * n_hops, "sum",
                                HEAD_DIM, hops, cluster_size)

            # ---- Stage 6: O projection; cross-head reduction via global atomicAdd
            for t in T.Pipelined(dim_per_block // tile_o, num_stages=num_stages):
                T.copy(Wo[rank * dim_per_block + t * tile_o, head * HEAD_DIM], s_wo)
                for i in T.Parallel(tile_o):
                    f_oa[i] = 0.0
                    for k in T.serial(HEAD_DIM):
                        f_oa[i] += T.Cast("float32", s_wo[i, k]) \
                            * T.Cast("float32", s_o_send[k])
                    T.atomic_add(Output[0, rank * dim_per_block + t * tile_o + i],
                                 T.Cast("float16", f_oa[i]))

    return kernel


_kernel_cache = {}


def get_kernel(**config):
    """Compile (or fetch from cache) the fused kernel for a config.

    Config keys: threads, cluster_size, tile_in, tile_s, tile_o, num_stages.
    """
    key = tuple(sorted(config.items()))
    if key not in _kernel_cache:
        _kernel_cache[key] = tilelang.compile(_build_kernel(**config), target="cuda")
    return _kernel_cache[key]


def llama_decoder_layer_sglang(input, residual, weight_qkv, weight_o,
                               k_cache, v_cache, rms_input_weight, eps, cos, sin,
                               **kernel_config):
    """Drop-in replacement for clusterfusion.llama_decoder_layer_sglang.

    input/residual: [1, 4096] half (residual is updated in place).
    weight_qkv: [12288, 4096] half, weight_o: [4096, 4096] half (Linear layout).
    k_cache/v_cache: [seq, 4096] half (NHD, historical tokens only).
    cos/sin: [64] fp32. Returns (output, residual, k_new[1,32,128], v_new[1,32,128]).
    Extra keyword args tune the kernel config (see get_kernel).
    """
    assert input.dtype == torch.float16 and input.is_cuda
    assert k_cache.shape == v_cache.shape and k_cache.is_contiguous() and v_cache.is_contiguous()
    output = torch.zeros_like(input)
    k_new = torch.empty(HEAD_NUM, HEAD_DIM, dtype=torch.float16, device=input.device)
    v_new = torch.empty_like(k_new)
    get_kernel(**kernel_config)(input, residual, weight_qkv, weight_o, k_cache,
                                v_cache, rms_input_weight, cos, sin, output,
                                k_new, v_new, float(eps))
    return output, residual, k_new.unsqueeze(0), v_new.unsqueeze(0)
