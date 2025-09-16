import functools
import logging
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from .cudnn import cudnn_batch_prefill_with_kv_cache
from .jit import (
    gen_batch_prefill_module,
    gen_customize_batch_prefill_module,
    gen_fmha_cutlass_sm100a_module,
    gen_single_prefill_module,
    get_batch_prefill_uri,
    get_single_prefill_uri,
    setup_cubin_loader,
    setup_metainfo_loader,
    trtllm_gen_fmha_module,
)
from .page import block_sparse_indices_to_vector_sparse_offsets, get_seq_lens
from .quantization import packbits, segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    determine_attention_backend,
    device_support_pdl,
    get_device_sm_count,
    is_float8,
    is_sm100a_supported,
    register_custom_op,
    register_fake_op,
)

@functools.cache
def get_batch_prefill_jit_module(module_name: str, jit_module: Any):
    plan_func = jit_module.plan.default
    ragged_run_func = jit_module.ragged_run.default
    paged_run_func = jit_module.paged_run.default

    # torch library for ragged_run
    @register_custom_op(
        f"flashinfer::{module_name}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "o",
            "maybe_lse",
        ),
    )
    def ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        ragged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_ragged_run")
    def _fake_ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # torch library for paged_run
    @register_custom_op(
        f"flashinfer::{module_name}_paged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        paged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_paged_run")
    def _fake_paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )