import dataclasses
import logging
import os
import re
import torch
from pathlib import Path
from typing import List, Optional, Union

import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

class ClusterFusionJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.CLUSTERFUSION_WORKSPACE_DIR / "clusterfusion_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("clusterfusion.jit: " + msg)


logger = ClusterFusionJITLogger("clusterfusion.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 90:
            raise RuntimeError("ClusterFusion requires sm90+")
        
sm90a_nvcc_flags = ["-gencode=arch=compute_90a,code=sm_90a"]

@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]
    is_class: bool = False
    needs_device_linking: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.CLUSTERFUSION_JIT_DIR / self.name / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return jit_env.CLUSTERFUSION_JIT_DIR / self.name / f"{self.name}.so"

    def get_library_path(self) -> Path:
        if self.aot_path.exists():
            return self.aot_path
        return self.jit_library_path

    @property
    def aot_path(self) -> Path:
        return jit_env.CLUSTERFUSION_AOT_DIR / self.name / f"{self.name}.so"

    def write_ninja(self) -> None:
        ninja_path = self.ninja_path
        ninja_path.parent.mkdir(parents=True, exist_ok=True)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
            needs_device_linking=self.needs_device_linking,
        )
        write_if_different(ninja_path, content)

    def build(self, verbose: bool) -> None:
        tmpdir = get_tmpdir()
        with FileLock(tmpdir / f"{self.name}.lock", thread_local=False):
            run_ninja(jit_env.CLUSTERFUSION_JIT_DIR, self.ninja_path, verbose)

    def build_and_load(self, class_name: str = None):
        if self.aot_path.exists():
            so_path = self.aot_path
        else:
            so_path = self.jit_library_path
            verbose = os.environ.get("CLUSTERFUSION_JIT_VERBOSE", "0") == "1"
            self.build(verbose)
        load_class = class_name is not None
        loader = torch.classes if load_class else torch.ops
        loader.load_library(so_path)
        if load_class:
            cls = torch._C._get_custom_class_python_wrapper(self.name, class_name)
            return cls
        return getattr(loader, self.name)
    
def gen_jit_spec(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    check_cuda_arch()
    verbose = os.environ.get("CLUSTERFUSION_JIT_VERBOSE", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        f"--threads={min(os.cpu_count() or 4, 32)}",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    if verbose:
        cuda_cflags += [
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    if extra_include_paths is not None:
        extra_include_paths = [Path(x) for x in extra_include_paths]
    sources = [Path(x) for x in sources]

    spec = JitSpec(
        name=name,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=extra_include_paths,
        needs_device_linking=needs_device_linking,
    )
    spec.write_ninja()
    return spec

def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.CLUSTERFUSION_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir