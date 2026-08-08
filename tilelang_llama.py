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


@T.macro
def _vec_allreduce_hop(rank, send, send2, recv, cbar, bar0, i, op, n, hop,
                       cluster_size):
    # One hop of the ring all-reduce: exchange `send` -> peer, then merge the
    # received partial into `send2`. T.copy_cluster is asynchronous — the TMA
    # engine reads `send` after issue — so we never write the in-flight source
    # buffer; the merge always lands in the other ping-pong buffer, which is
    # only safe to rewrite once the previous hop's cluster_sync confirmed the
    # peer consumed it. The hop index and barrier slot are compile-time
    # constants; `rank`/`dst_block` are traced runtime values.
    T.sync_threads()
    T.fence_proxy_async()
    T.copy_cluster(send, recv, dst_block=(rank + hop) % cluster_size,
                   remote_barrier=cbar[bar0 + i])
    T.mbarrier_wait_parity(cbar[bar0 + i], 0)
    T.fence_proxy_async()
    # Explicit thread-strided merge instead of T.Parallel (layout-independent
    # thread mapping). Note: the intermittent wrong k/v values persist even
    # with full-thread cluster_sync and with the SIMT copy fallback — the
    # corruption is in tilelang 0.1.13's cluster synchronization/copy path
    # itself, not in this merge loop.
    tid = T.get_thread_binding()
    n_threads = T.get_thread_extent(0)
    for j in T.serial(T.ceildiv(n, n_threads)):
        if j * n_threads + tid < n:
            if op == "sum":
                send2[j * n_threads + tid] = send[j * n_threads + tid] + recv[j * n_threads + tid]
            else:
                send2[j * n_threads + tid] = T.max(send[j * n_threads + tid], recv[j * n_threads + tid])
    T.sync_threads()
    T.cluster_sync()


@T.macro
def _vec_copy_back(dst, src, n):
    # Odd hop counts leave the full reduction in the secondary buffer; fold it
    # back into the canonical `send` buffer after the final cluster_sync.
    tid = T.get_thread_binding()
    n_threads = T.get_thread_extent(0)
    for j in T.serial(T.ceildiv(n, n_threads)):
        if j * n_threads + tid < n:
            dst[j * n_threads + tid] = src[j * n_threads + tid]
    T.sync_threads()


def _ring_allreduce_vec(rank, send, send2, recv, cbar, bar0, op, n, hops,
                        cluster_size):
    # Ring all-reduce on a half vector; `send` holds the accumulated partial
    # and ends up holding the full reduction on every CTA. The hop loop is a
    # compile-time unroll (hop values and barrier slots are Python constants);
    # each hop copies from one buffer and merges into the other.
    for i, hop in enumerate(hops):
        if i % 2 == 0:
            _vec_allreduce_hop(rank, send, send2, recv, cbar, bar0, i, op, n,
                               hop, cluster_size)
        else:
            _vec_allreduce_hop(rank, send2, send, recv, cbar, bar0, i, op, n,
                               hop, cluster_size)
    if len(hops) % 2 == 1:
        _vec_copy_back(send, send2, n)


@T.macro
def _scalar_allreduce_hop(rank, send, send2, recv, cbar, bar0, i, op, hop,
                          cluster_size):
    # Same protocol on one float (packed in a 16B buffer to keep the
    # cp.async.bulk fast path); same ping-pong source/merge split as the
    # vector version.
    T.sync_threads()
    T.fence_proxy_async()
    T.copy_cluster(send, recv, dst_block=(rank + hop) % cluster_size,
                   remote_barrier=cbar[bar0 + i])
    T.mbarrier_wait_parity(cbar[bar0 + i], 0)
    T.fence_proxy_async()
    if T.get_thread_binding() == 0:
        if op == "sum":
            send2[0] = send[0] + recv[0]
        else:
            send2[0] = T.max(send[0], recv[0])
    T.sync_threads()
    T.cluster_sync()


@T.macro
def _scalar_copy_back(dst, src):
    # Fold odd-hop parity back into the canonical `send` buffer.
    if T.get_thread_binding() == 0:
        dst[0] = src[0]
    T.sync_threads()


def _ring_allreduce_scalar(rank, send, send2, recv, cbar, bar0, op, hops,
                           cluster_size):
    # Same protocol on one float (packed in a 16B buffer to keep the
    # cp.async.bulk fast path). Hop loop is a compile-time unroll; each hop
    # copies from one buffer and merges into the other.
    for i, hop in enumerate(hops):
        if i % 2 == 0:
            _scalar_allreduce_hop(rank, send, send2, recv, cbar, bar0, i, op,
                                  hop, cluster_size)
        else:
            _scalar_allreduce_hop(rank, send2, send, recv, cbar, bar0, i, op,
                                  hop, cluster_size)
    if len(hops) % 2 == 1:
        _scalar_copy_back(send, send2)


def _estimate_smem_bytes(threads, cluster_size, tile_in, tile_s, tile_o, num_stages):
    dim_per_block = HIDDEN // cluster_size
    n_hops = max(0, cluster_size.bit_length() - 1)
    total = (
        dim_per_block * 4                              # s_x (f32)
        + dim_per_block * 2                            # s_xn (half)
        + HEAD_DIM * tile_in * 2 * num_stages          # s_w (qkv weight tiles)
        + 3 * HEAD_DIM * 2 * 3                         # s_qkv + s_qkv_recv + s_qkv2
        + tile_s * HEAD_DIM * 2 * 2 * num_stages       # s_k + s_v
        + tile_s * 4                                   # s_p
        + tile_o * HEAD_DIM * 2 * num_stages           # s_wo
        + HEAD_DIM * 2 * 3                             # s_o_send + s_o_recv + s_o_send2
        + dim_per_block * 2                            # s_out projection staging
        + 80                                           # scalar shared buffers + s_ssend2
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
            s_qkv2 = T.alloc_shared((3 * HEAD_DIM,), "float16")
            s_k = T.alloc_shared((tile_s, HEAD_DIM), "float16")
            s_v = T.alloc_shared((tile_s, HEAD_DIM), "float16")
            s_p = T.alloc_shared((tile_s,), "float32")
            s_wo = T.alloc_shared((tile_o, HEAD_DIM), "float16")
            s_o_send = T.alloc_shared((HEAD_DIM,), "float16")
            s_o_recv = T.alloc_shared((HEAD_DIM,), "float16")
            s_o_send2 = T.alloc_shared((HEAD_DIM,), "float16")
            s_out = T.alloc_shared((dim_per_block,), "float16")
            s_ssend = T.alloc_shared((4,), "float32")
            s_srecv = T.alloc_shared((4,), "float32")
            s_ssend2 = T.alloc_shared((4,), "float32")
            s_ssq = T.alloc_shared((1,), "float32")
            s_rms = T.alloc_shared((1,), "float32")
            s_alpha = T.alloc_shared((1,), "float32")
            s_ct = T.alloc_shared((1,), "float32")
            s_pct = T.alloc_shared((1,), "float32")
            # 5 collectives x n_hops barriers each: RMS, QKV, softmax max/sum, ATTN
            # (max(1, ...) keeps a dummy barrier for cluster_size == 1)
            cbar = T.alloc_cluster_barrier([1] * max(1, 5 * n_hops))

            f_acc = T.alloc_fragment((HEAD_DIM,), "float32")
            f_scores = T.alloc_fragment((tile_s,), "float32")
            f_prod = T.alloc_fragment((tile_s, HEAD_DIM), "float32")
            f_p = T.alloc_fragment((tile_s,), "float32")
            f_o = T.alloc_fragment((HEAD_DIM,), "float32")
            f_pv = T.alloc_fragment((HEAD_DIM,), "float32")
            f_ct = T.alloc_fragment((1, HEAD_DIM), "float32")
            f_oa = T.alloc_fragment((tile_o,), "float32")
            f_prod_o = T.alloc_fragment((tile_o, HEAD_DIM), "float32")
            # Loop-carried online-softmax state. Must be fragments (registers),
            # not shared buffers: T.Pipelined multi-versions shared buffers that
            # are stored inside the loop, and reads after the loop then fail
            # ("Versioned buffer load escaped pipeline stage context").
            f_tm = T.alloc_fragment((1,), "float32")
            f_tl = T.alloc_fragment((1,), "float32")
            f_m = T.alloc_fragment((1,), "float32")
            f_l = T.alloc_fragment((1,), "float32")
            f_alpha = T.alloc_fragment((1,), "float32")
            f_pct = T.alloc_fragment((1,), "float32")

            # ---- Stage 1: fused residual add + RMSNorm (cluster-reduced sum of squares)
            # h = Input + Residual is computed by every head cluster on the same
            # residual slice. Writing it back here would race across clusters
            # (a peer can read the already-updated residual and fold it into h
            # again), so the global residual update happens once on the host
            # after the launch (see llama_decoder_layer_sglang).
            for i in T.Parallel(dim_per_block):
                h = T.Cast("float32", Input[0, rank * dim_per_block + i]) \
                    + T.Cast("float32", Residual[0, rank * dim_per_block + i])
                s_x[i] = h
            T.sync_threads()
            # Deterministic sum-of-squares. The cross-thread tl.reduce_sum /
            # AllReduce is nondeterministic in this kernel (occasionally wrong
            # partial sums -> wrong RMS -> intermittent wrong k/v/output).
            # Each thread accumulates its own slice and one thread serially
            # sums the partials; n_threads is the actual launch thread count.
            n_threads = T.get_thread_extent(0)
            s_part = T.alloc_shared((256,), "float32")
            f_loc = T.alloc_fragment((1,), "float32")
            f_loc[0] = 0.0
            for i in T.serial(dim_per_block // n_threads):
                f_loc[0] += s_x[tid * (dim_per_block // n_threads) + i] * \
                    s_x[tid * (dim_per_block // n_threads) + i]
            s_part[tid] = f_loc[0]
            T.sync_threads()
            if tid == 0:
                s_ssq[0] = 0.0
                for i in T.serial(n_threads):
                    s_ssq[0] += s_part[i]
            T.sync_threads()
            if tid == 0:
                s_ssend[0] = s_ssq[0]
            _ring_allreduce_scalar(rank, s_ssend, s_ssend2, s_srecv, cbar, 0,
                                   "sum", hops, cluster_size)
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
            _ring_allreduce_vec(rank, s_qkv, s_qkv2, s_qkv_recv, cbar, n_hops,
                                "sum", 3 * HEAD_DIM, hops, cluster_size)

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
            T.fill(f_m, neg_inf)
            T.fill(f_l, 0.0)
            for t in T.Pipelined(T.ceildiv(kv_per_block, tile_s), num_stages=num_stages):
                T.copy(KCache[rank * kv_per_block + t * tile_s, head * HEAD_DIM], s_k)
                T.copy(VCache[rank * kv_per_block + t * tile_s, head * HEAD_DIM], s_v)
                # QK^T as parallel elementwise products + one reduction: the old
                # per-score 128-step serial FMA chain used only 64 of the 128
                # consumer threads and was the dominant attention cost (~50 us
                # at kv=4096). Spreading the products over all threads through a
                # (tile_s, HEAD_DIM) fragment and letting T.reduce_sum handle the
                # cross-thread combine made the whole kernel ~1.3x faster.
                for r, d in T.Parallel(tile_s, HEAD_DIM):
                    f_prod[r, d] = T.Cast("float32", s_qkv[d]) \
                        * T.Cast("float32", s_k[r, d])
                T.reduce_sum(f_prod, f_scores, dim=1)
                for r in T.Parallel(tile_s):
                    f_scores[r] = T.if_then_else(
                        rank * kv_per_block + t * tile_s + r < seqlen,
                        f_scores[r] * QK_SCALE, neg_inf)
                # Cross-thread reduce into replicated (1,) fragments: every
                # thread ends up with the tile max/sum (tl::AllReduce internally),
                # so m/l/alpha live in registers across iterations.
                T.reduce_max(f_scores, f_tm, dim=0)
                # alpha rescales the running o/l when the tile max advances;
                # the equality guard also covers an all-(-inf) tile (alpha=1).
                f_alpha[0] = T.if_then_else(
                    f_tm[0] == f_m[0], 1.0,
                    T.exp2(f_m[0] - T.max(f_m[0], f_tm[0])))
                f_m[0] = T.max(f_m[0], f_tm[0])
                for r in T.Parallel(tile_s):
                    f_p[r] = T.if_then_else(f_scores[r] == neg_inf, 0.0,
                                            T.exp2(f_scores[r] - f_m[0]))
                T.reduce_sum(f_p, f_tl, dim=0)
                T.copy(f_p, s_p)
                f_l[0] = f_l[0] * f_alpha[0] + f_tl[0]
                for d in T.Parallel(HEAD_DIM):
                    f_pv[d] = 0.0
                    for r in T.serial(tile_s):
                        f_pv[d] += s_p[r] * T.Cast("float32", s_v[r, d])
                    f_o[d] = f_o[d] * f_alpha[0] + f_pv[d]

            # current token's k/v join the softmax exactly once (rank 0)
            if rank == 0:
                for d in T.Parallel(HEAD_DIM):
                    f_ct[0, d] = T.Cast("float32", s_qkv[d]) \
                        * T.Cast("float32", s_qkv[HEAD_DIM + d])
                T.reduce_sum(f_ct, s_ct, dim=1)
                T.sync_threads()
                sc = s_ct[0] * QK_SCALE
                f_alpha[0] = T.if_then_else(
                    sc == f_m[0], 1.0,
                    T.exp2(f_m[0] - T.max(f_m[0], sc)))
                f_pct[0] = T.exp2(sc - T.max(f_m[0], sc))
                f_m[0] = T.max(f_m[0], sc)
                f_l[0] = f_l[0] * f_alpha[0] + f_pct[0]
                T.sync_threads()
                for d in T.Parallel(HEAD_DIM):
                    f_o[d] = f_o[d] * f_alpha[0] \
                        + f_pct[0] * T.Cast("float32", s_qkv[2 * HEAD_DIM + d])

            # ---- Stage 5: cluster-level softmax merge (max, rescale, sum, normalize)
            T.sync_threads()
            if tid == 0:
                s_ssend[0] = f_m[0]
            _ring_allreduce_scalar(rank, s_ssend, s_ssend2, s_srecv, cbar,
                                   2 * n_hops, "max", hops, cluster_size)
            if tid == 0:
                s_alpha[0] = T.exp2(f_m[0] - s_ssend[0])  # 0 if this CTA had no valid row
            T.sync_threads()
            for d in T.Parallel(HEAD_DIM):
                f_o[d] = f_o[d] * s_alpha[0]
            T.sync_threads()
            if tid == 0:
                # l must be rescaled by this rank's alpha before the cluster
                # sum; f_o above already carries the same rescale.
                s_ssend[0] = f_l[0] * s_alpha[0]
            _ring_allreduce_scalar(rank, s_ssend, s_ssend2, s_srecv, cbar,
                                   3 * n_hops, "sum", hops, cluster_size)
            for d in T.Parallel(HEAD_DIM):
                s_o_send[d] = T.Cast("float16", f_o[d] / s_ssend[0])
            T.sync_threads()
            # ClusterReduce #2: full head output on every CTA
            _ring_allreduce_vec(rank, s_o_send, s_o_send2, s_o_recv, cbar,
                                4 * n_hops, "sum", HEAD_DIM, hops, cluster_size)

            # ---- Stage 6: O projection; cross-head reduction via global atomicAdd
            # T.atomic_add must see exactly one contributing thread per output
            # element. Inside a T.Parallel(tile_o) loop tile_o < threads, the
            # fragment layout is replicated across thread groups and the atomic
            # would be executed once per replica. Stage the per-tile projection
            # in shared memory with a serial loop, then do one distinct
            # element-per-thread atomic pass over the whole slice.
            for t in T.serial(dim_per_block // tile_o):
                T.copy(Wo[rank * dim_per_block + t * tile_o, head * HEAD_DIM], s_wo)
                # Same parallel-products treatment as QK^T above: the serial
                # per-row FMA chain was latency-bound on the Wo shared loads; a
                # (tile_o, HEAD_DIM) fragment + reduce_sum spreads it over all
                # threads.
                for i, k in T.Parallel(tile_o, HEAD_DIM):
                    f_prod_o[i, k] = T.Cast("float32", s_wo[i, k]) \
                        * T.Cast("float32", s_o_send[k])
                T.reduce_sum(f_prod_o, f_oa, dim=1)
                for i in T.Parallel(tile_o):
                    s_out[t * tile_o + i] = T.Cast("float16", f_oa[i])
            T.sync_threads()
            for i in T.Parallel(dim_per_block):
                T.atomic_add(Output[0, rank * dim_per_block + i], s_out[i])

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
    # Every head cluster redundantly computes h = input + residual for its own
    # slice, so the global residual update is done here exactly once, after all
    # kernel reads have completed (same CUDA stream).
    residual.copy_((input.float() + residual.float()).half())
    return output, residual, k_new.unsqueeze(0), v_new.unsqueeze(0)
