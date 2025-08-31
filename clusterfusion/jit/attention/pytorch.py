import os
from typing import List

import jinja2
import torch

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from ..utils import (
    dtype_map,
    mask_mode_literal,
    pos_encoding_mode_literal,
    write_if_different,
)
from .utils import generate_additional_params

def gen_customize_batch_decode_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
) -> JitSpec:
    gen_directory = jit_env.CLUSTERFUSION_GEN_SRC_DIR / uri
    (additional_params_decl, additional_func_params, additional_params_setter) = (
        generate_additional_params(
            additional_tensor_names,
            additional_tensor_dtypes,
            additional_scalar_names,
            additional_scalar_dtypes,
        )
    )

    kwargs = {
        "additional_params_decl": additional_params_decl,
        "additional_func_params": additional_func_params,
        "additional_params_setter": additional_params_setter,
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }

    with open(jit_env.CLUSTERFUSION_CSRC_DIR / "batch_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(jit_env.CLUSTERFUSION_CSRC_DIR / "batch_decode_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    generated_inc_str = config_templ.render(
        **kwargs,
    )

    source_paths = []

    dest_path = gen_directory / "batch_decode_kernel.cu"
    source_paths.append(dest_path)
    source = kernel_inst_templ.render(
        **kwargs,
    )
    write_if_different(dest_path, source)

    for filename in [
        "batch_decode.cu",
        "batch_decode_jit_pybind.cu",
    ]:
        src_path = jit_env.CLUSTERFUSION_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return gen_jit_spec(uri, source_paths)

def gen_customize_batch_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    fp8_enabled: bool = False,
) -> JitSpec:
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = jit_env.CLUSTERFUSION_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(
            jit_env.CLUSTERFUSION_CSRC_DIR / "batch_prefill_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        with open(
            jit_env.CLUSTERFUSION_CSRC_DIR / "batch_prefill_paged_kernel_inst.jinja"
        ) as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(
            jit_env.CLUSTERFUSION_CSRC_DIR / "batch_prefill_ragged_kernel_inst.jinja"
        ) as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(
            **kwargs,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for mask_mode in [0, 1, 2, 3]:
            dest_path = (
                gen_directory / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
            )
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            dest_path = (
                gen_directory / f"batch_prefill_ragged_kernel_mask_{mask_mode}.cu"
            )
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "batch_prefill.cu",
            "batch_prefill_jit_pybind.cu",
        ]:
            src_path = jit_env.CLUSTERFUSION_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return gen_jit_spec(uri, source_paths)
    elif backend == "fa3":
        gen_directory = jit_env.CLUSTERFUSION_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
                is_sm90_template=True,
            )
        )

        _file_config = "batch_prefill_sm90_customize_config.jinja"
        if fp8_enabled:
            _file_paged_kernel_inst = "batch_prefill_fp8_paged_sm90_kernel_inst.jinja"
            _file_ragged_kernel_inst = "batch_prefill_fp8_ragged_sm90_kernel_inst.jinja"
            _file_csrc = "batch_prefill_fp8_sm90.cu"
        else:
            _file_paged_kernel_inst = "batch_prefill_paged_sm90_kernel_inst.jinja"
            _file_ragged_kernel_inst = "batch_prefill_ragged_sm90_kernel_inst.jinja"
            _file_csrc = "batch_prefill_sm90.cu"

        with open(jit_env.CLUSTERFUSION_CSRC_DIR / _file_config) as f:
            config_templ = jinja2.Template(f.read())

        with open(jit_env.CLUSTERFUSION_CSRC_DIR / _file_paged_kernel_inst) as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(jit_env.CLUSTERFUSION_CSRC_DIR / _file_ragged_kernel_inst) as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }
        generated_inc_str = config_templ.render(**kwargs)

        source_paths = []
        for mask_mode in [0, 1, 2, 3]:
            filename = f"batch_prefill_paged_sm90_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            filename = f"batch_prefill_ragged_sm90_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            _file_csrc,
            "batch_prefill_sm90_jit_pybind.cu",
        ]:
            src_path = jit_env.CLUSTERFUSION_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_sm90_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return gen_jit_spec(
            uri,
            source_paths,
            extra_cuda_cflags=sm90a_nvcc_flags,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")

def cudnn_fmha_gen_module():
    return gen_jit_spec(
        "fmha_cudnn_gen",
        [jit_env.CLUSTERFUSION_CSRC_DIR / "cudnn_sdpa_kernel_launcher.cu"],
        extra_ldflags=["-lcuda"],
    )
