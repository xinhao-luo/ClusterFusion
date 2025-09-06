import functools
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from .jit import (
    gen_batch_decode_module,
    gen_customize_batch_decode_module,
    gen_customize_batch_prefill_module,
    get_batch_decode_uri,
    get_batch_prefill_uri,
)
from .page import get_seq_lens
from .prefill import (
    get_batch_prefill_jit_module,
    get_batch_prefill_module,
)
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    device_support_pdl,
    get_device_sm_count,
    register_custom_op,
    register_fake_op,
)

@functools.cache
def get_batch_decode_jit_module(module_name: str, jit_module: Any):
    plan_func = jit_module.plan.default
    run_func = jit_module.run.default

    @register_custom_op(
        f"flashinfer::{module_name}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        *args,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            enable_pdl,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        *args,
    ) -> None:
        pass

    return SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )

@functools.cache
def get_batch_decode_module(*args):
    uri = get_batch_decode_uri(*args)
    mod = gen_batch_decode_module(*args).build_and_load()
    plan_func = mod.plan.default
    run_func = mod.run.default

    # torch library for batch_decode_with_paged_kv_cache_run

    @register_custom_op(
        f"flashinfer::{uri}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            enable_pdl,
            alibi_slopes,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )

class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with paged kv-cache (first proposed in
    `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        paged_kv_indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``clusterfusion``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            if use_tensor_cores:
                self._jit_module = get_batch_prefill_jit_module(
                    jit_args[0],
                    gen_customize_batch_prefill_module(
                        "fa2", *jit_args
                    ).build_and_load(),
                )
            else:
                self._jit_module = get_batch_decode_jit_module(
                    jit_args[0],
                    gen_customize_batch_decode_module(*jit_args).build_and_load(),
                )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._kv_lens_buffer: Optional[torch.Tensor] = None
        
        # TODO(Chiheng): add clusterfusion kv_lens_buffer limit

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        # TODO: comfirm that clusterfusion uses tensor cores
        self._use_tensor_cores = use_tensor_cores or backend == "clusterfusion"
        self._use_cuda_graph = use_cuda_graph

        if use_tensor_cores:
            if use_cuda_graph:
                # NOTE(Zihao): if once created, no need to update it in plan/run
                self._qo_indptr_buf = torch.arange(
                    self._fixed_batch_size + 1,
                    dtype=torch.int32,
                    device=float_workspace_buffer.device,
                )
        self._backend = backend

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Plan batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        seq_lens: Optional[torch.Tensor]
            A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``.
        block_tables: Optional[torch.Tensor]
            A uint32 2D tensor indicating the block table of each prompt. shape: ``[batch_size, max_num_blocks_per_seq]``.


        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(indices)].copy_(
                indices, non_blocking=(indices.device == self.device) and non_blocking
            )
        else:
            self._paged_kv_indptr_buf = indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

        indptr_host = indptr.to("cpu")
        last_page_len_host = last_page_len.to("cpu")

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        self._batch_size = batch_size
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._block_tables: Optional[torch.Tensor] = block_tables
        self._max_kv_len: Optional[int] = None

        if seq_lens is None:
            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
        else:
            kv_lens_arr_host = seq_lens.cpu()
        # TODO(Chiheng): switch to clusterfusion
        if self._backend == "clusterfusion":
            assert self._kv_layout == "NHD", "ClusterFusion backend only supports NHD layout"
            self._max_kv_len = max(kv_lens_arr_host).item()
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host, non_blocking=non_blocking
            )
            if self._block_tables is None:
                blocks_per_seq = [
                    (seq_len + page_size - 1) // page_size
                    for seq_len in kv_lens_arr_host
                ]
                max_num_blocks_per_seq = max(blocks_per_seq)
                self._block_tables = torch.zeros(
                    (batch_size, max_num_blocks_per_seq),
                    dtype=torch.int,
                    device=self.device,
                )
                block_id = indptr[0]
                for i in range(batch_size):
                    num_blocks_needed = blocks_per_seq[i]
                    self._block_tables[i, :num_blocks_needed] = (
                        self._paged_kv_indices_buf[
                            block_id : block_id + num_blocks_needed
                        ]
                    )
                    block_id += num_blocks_needed
            self._cached_module = get_trtllm_gen_decode_module(
                q_data_type,
                kv_data_type,
                q_data_type,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )
            self._plan_info = self._cached_module.plan()  # None
        elif self.use_tensor_cores:
            self._max_kv_len = max(kv_lens_arr_host).item()
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_prefill_module(
                    "fa2",
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                indptr_host,
                kv_lens_arr_host,
                batch_size,  # total_num_rows
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                head_dim,
                head_dim,
                False,  # causal
            )
        else:
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_decode_module(
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                indptr_host,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                window_left,
                logits_soft_cap,
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: this function is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q, paged_kv_cache, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
        )

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        *args
            Additional arguments for the custom kernel.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention scores, defaults to ``False``.
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * attention output, shape: ``[batch_size, num_qo_heads, head_dim]``
            * logsumexp of attention scores, shape: ``[batch_size, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        if self._kv_layout == "NHD":
            page_size = k_cache.shape[1]
            stride_n = k_cache.stride(1)
        else:
            page_size = k_cache.shape[2]
            stride_n = k_cache.stride(2)
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )

        pos_encoding_mode = self._pos_encoding_mode
        window_left = self._window_left if window_left is None else window_left
        # TODO(Chiheng): figure out what the following comment means
        if self._backend != "trtllm-gen":
            # NOTE(Siyuan): since window_left is appeared in the plan function, we need to make sure it is the same as the one in the plan function.
            # Remove this check if the backend supports dynamic window_left.
            assert window_left == self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q)
        else:
            _check_shape_dtype_device(out, q.shape, q.dtype, q.device, "out")

        if self.use_tensor_cores:
            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._qo_indptr_buf,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                MaskMode.NON_CAUSAL.value,
                TensorLayout[self._kv_layout].value,
                window_left,
                enable_pdl,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    None,  # packed_custom_mask
                    None,  # mask_indptr_buf
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    None,  # maybe_prefix_len_ptr
                    None,  # maybe_token_pos_in_items_ptr
                    None,  # maybe_max_item_len_ptr
                    logits_soft_cap,
                    sm_scale,
                    None,  # scale_q, not supported yet
                    None,  # scale_k
                    None,  # scale_v
                    rope_scale,
                    rope_theta,
                    0,  # token_pos_in_items_len
                    paged_kv_cache,
                    self._num_qo_heads,
                    self._num_kv_heads,
                    self._block_tables,
                    self._kv_lens_buffer,
                    page_size,
                    self._max_kv_len,
                ]

            self._cached_module.paged_run(*run_args)
        else:
            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                window_left,
                enable_pdl,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    logits_soft_cap,
                    sm_scale,
                    rope_scale,
                    rope_theta,
                ]

            self._cached_module.run(*run_args)
        if v_scale is not None:
            # TODO(Zihao): fused into kernel
            if out.itemsize == 1:
                out = (out.to(float) * v_scale).to(out.dtype)
            else:
                out *= v_scale

        return (out, lse) if return_lse else out

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: this function is deprecated, please use :meth:`run_return_lse` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q,
            paged_kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass

# TODO(Chiheng): switch to clusterfusion
class TrtllmGenDecodeModule:
    def _paged_run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        bmm1_scale: float,  # todo(Yingyi): add dynamic scale tensor later
        bmm2_scale: float,
        window_left: int = -1,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(query)
        if self._sm_count is None:
            self._sm_count = get_device_sm_count(query.device)
        self._op.trtllm_paged_attention_decode(
            out,
            query.unsqueeze(
                1
            ),  # [B, 1, H, D], no MTP here so second dim is 1 # todo(Yingyi): add MTP??
            kv_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_seq_len,
            bmm1_scale,
            bmm2_scale,
            window_left,
            self._sm_count,
        )
        return out

    def _plan(self, *args, **kwargs):
        pass

    def __init__(self):
        self._sm_count: Optional[int] = None
        self._mod = trtllm_gen_fmha_module()
        self._op = self._mod.build_and_load()
        from flashinfer.jit.cubin_loader import (
            setup_cubin_loader,
            setup_metainfo_loader,
        )

        setup_cubin_loader(self._mod.get_library_path())
        setup_metainfo_loader(self._mod.get_library_path())

# TODO(Chiheng): switch to clusterfusion
@functools.cache
def get_trtllm_gen_decode_module(*args):
    uri = get_batch_prefill_uri("trtllm-gen", *args)
    module = TrtllmGenDecodeModule()

    @register_custom_op(
        f"flashinfer::{uri}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
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
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        paged_kv_cache: Optional[torch.Tensor] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_kv_len: Optional[int] = None,
    ) -> None:
        assert maybe_lse is None
        assert paged_kv_cache is not None
        assert num_qo_heads is not None
        assert num_kv_heads is not None
        assert block_tables is not None
        assert kv_lens_buffer is not None
        assert page_size is not None
        assert max_kv_len is not None
        o = module._paged_run(
            q.contiguous(),  # NOTE(Siyuan): without contiguous, the result is incorrect
            paged_kv_cache,
            int_workspace_buffer,
            block_tables,
            kv_lens_buffer,
            max_kv_len,
            sm_scale,
            1.0,  # NOTE(Siyuan): update this to expose bmm2 scale
            window_left,
            out=o,
        )

    @register_fake_op(f"flashinfer::{uri}_paged_run")
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
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        paged_kv_cache: Optional[torch.Tensor] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_kv_len: Optional[int] = None,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=module._plan,
        paged_run=paged_run,
    )