# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import itertools
from typing import Dict, List, Optional, Tuple, Union

import pytest
import torch

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _attention_decode,
    _attention_prefill,
    _attention_prefill_ragged,
    _compact_kv_copy,
    _copy_single_page,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _merge_state_inplace,
    llama_rope_with_position_map,
)
from tvm.runtime import ShapeTuple

reserved_nseq = 32
maximum_total_seq_length = 16384
prefill_chunk_size = 16384
page_size = 16
num_layers = 32
num_qo_heads = 32
num_kv_heads = 32
head_dim = None
sm_scale = None
rope_scale = 1.0
rope_theta = 1e4
rope_scaling = {}
dtype = None
dtype_torch = None
device = tvm.cuda()
device_torch = torch.device("cuda")
fclear = None
fadd_sequence = None
fbegin_forward = None
fend_forward = None
fattention_with_fuse_qkv = None

ftranspose_append = None
fcopy_cache = None
fattn_prefill = None
fattn_decode = None
fattn_prefill_ragged = None
fmerge_state = None
fsplit_rotary = None
fcopy_single_page = None
fcompact_copy = None


def set_global_func(head_dim, dtype):
    global fclear, fadd_sequence
    global fbegin_forward, fend_forward
    global fattention_with_fuse_qkv
    global ftranspose_append, fcopy_cache, fattn_prefill, fattn_decode
    global fattn_prefill_ragged
    global fmerge_state, fsplit_rotary, fcopy_single_page, fcompact_copy

    fclear = tvm.get_global_func("vm.builtin.kv_state_clear")
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fattention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_attention_with_fused_qkv"
    )

    target = tvm.target.Target.from_device(device)
    builts = []
    for tir_func in [
        _kv_cache_transpose_append(num_kv_heads, head_dim, dtype),
        _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype),
        _attention_prefill(
            num_kv_heads, num_qo_heads, head_dim, dtype, False, rope_scaling, target
        ),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, False, rope_scaling, target),
        _attention_prefill_ragged(
            num_kv_heads, num_qo_heads, head_dim, head_dim, dtype, rope_scaling, target
        ),
        _merge_state_inplace(num_qo_heads, head_dim, dtype, target),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype, rope_scaling
        ),
        _copy_single_page(num_kv_heads, page_size, head_dim, dtype, target),
        _compact_kv_copy(num_kv_heads, head_dim, dtype, target),
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.tir.build(mod["main"], target=target)
        builts.append(f.entry_func)

    (
        ftranspose_append,
        fcopy_cache,
        fattn_prefill,
        fattn_decode,
        fattn_prefill_ragged,
        fmerge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcompact_copy,
    ) = builts


def create_kv_cache(head_dim, dtype, rope_mode):
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    cache = fcreate(
        tvm.runtime.ShapeTuple(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size,
                0
            ]
        ),
        tvm.runtime.ShapeTuple([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,  # v_head_dim
        tvm.runtime.ShapeTuple([int(AttnKind.MHA) for _ in range(num_layers)]),
        False,  # enable_kv_transfer
        rope_mode,
        rope_scale,
        rope_theta,
        None,  # rope_ext_factors
        tvm.nd.empty((), dtype, device=device),
        ftranspose_append,
        None,  # f_transpose_append_mla
        ["tir", fattn_prefill_ragged],
        ["tir", fattn_prefill],
        ["tir", fattn_decode],
        [],
        [],
        [],
        [],
        [],  # f_mla_prefill
        [fmerge_state],
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
    )
    return cache


@pytest.fixture(
    params=itertools.chain(
        itertools.product(
            [128],
            ["float16"],
            [RopeMode.NONE],
        ),
    )
)
def kv_cache_and_config(request):
    global head_dim, sm_scale, dtype, dtype_torch
    head_dim, dtype, rope_mode = request.param
    dtype_torch = getattr(torch, dtype)
    sm_scale = head_dim ** (-0.5)
    set_global_func(head_dim, dtype)
    return create_kv_cache(*request.param), rope_mode

def apply_attention(
    kv_cache,
    rope_mode: RopeMode,
    batch: List[Tuple[Union[int, Tuple[int, int, int]], int]],
    cached_k: Dict[int, torch.Tensor],
    cached_v: Dict[int, torch.Tensor],
) -> None:
    seq_ids = []
    append_lengths = []
    for i, (seq_id, append_length) in enumerate(batch):
        seq_ids.append(seq_id)
        append_lengths.append(append_length)
        if seq_id not in cached_k:
            fadd_sequence(kv_cache, seq_id)
            cached_k[seq_id] = torch.zeros(
                (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
            )
            cached_v[seq_id] = torch.zeros(
                (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
            )

    fbegin_forward(
        kv_cache,
        ShapeTuple(seq_ids),
        ShapeTuple(append_lengths),
        (
            None
        ),
    )

    global_new_q = torch.zeros(
        (num_layers, 0, num_qo_heads, head_dim), dtype=dtype_torch, device=device_torch
    )
    global_new_k = torch.zeros(
        (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
    )
    global_new_v = torch.zeros(
        (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
    )

    q_array = []
    for i, (seq_id, append_length) in enumerate(batch):
        new_q = torch.rand(
            num_layers,
            append_length,
            num_qo_heads,
            head_dim,
            dtype=dtype_torch,
            device=device_torch,
        )
        new_k = torch.rand(
            num_layers,
            append_length,
            num_kv_heads,
            head_dim,
            dtype=dtype_torch,
            device=device_torch,
        )
        new_v = torch.rand(
            num_layers,
            append_length,
            num_kv_heads,
            head_dim,
            dtype=dtype_torch,
            device=device_torch,
        )
        new_q = new_q * 2 - 1
        new_k = new_k * 2 - 1
        new_v = new_v * 2 - 1
        q_array.append(new_q)

        cached_k[seq_id] = torch.cat(
            [
                cached_k[seq_id],
                torch.stack(
                    [
                        (
                            new_k[l]
                        )
                        for l in range(num_layers)
                    ],
                    dim=0,
                ),
            ],
            dim=1,
        )
        cached_v[seq_id] = torch.cat([cached_v[seq_id], new_v], dim=1)
        global_new_q = torch.cat([global_new_q, new_q], dim=1)
        global_new_k = torch.cat([global_new_k, new_k], dim=1)
        global_new_v = torch.cat([global_new_v, new_v], dim=1)

    for layer_id in range(num_layers):
        queries_np = global_new_q[layer_id]
        keys_np = global_new_k[layer_id]
        values_np = global_new_v[layer_id]
        qkv = tvm.nd.array(torch.cat([queries_np, keys_np, values_np], dim=1).cpu().numpy(), device)
        outputs = tvm.nd.empty(queries_np.shape, dtype, device=device)
        fattention_with_fuse_qkv(kv_cache, layer_id, sm_scale, qkv, outputs)
    
    fend_forward(kv_cache)

@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_prefill_and_decode(kv_cache_and_config):
    kv_cache, rope_mode = kv_cache_and_config
    fclear(kv_cache)

    # Prefill.
    operation_seq = [[(0, 4096 - 1)]]
    # Decode
    operation_seq += [[(0, 1)]]

    cached_k = {}
    cached_v = {}
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)

if __name__ == "__main__":
    HEAD_DIMS = [128]
    DTYPES = ["float16"]
    ROPE_MODES = [RopeMode.NONE]
    for head_dim, dtype, rope_mode in itertools.product(
        HEAD_DIMS, DTYPES, ROPE_MODES
    ):
        dtype_torch = getattr(torch, dtype)
        sm_scale = head_dim ** (-0.5)
        set_global_func(head_dim, dtype)
        cache = create_kv_cache(head_dim, dtype, rope_mode)
        cache_and_config = (cache, rope_mode)
        test_paged_attention_kv_cache_prefill_and_decode(cache_and_config)