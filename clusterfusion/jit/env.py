import os
import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags

CLUSTERFUSION_BASE_DIR = pathlib.Path(
    os.getenv("CLUSTERFUSION_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

CLUSTERFUSION_CACHE_DIR = CLUSTERFUSION_BASE_DIR / ".cache" / "CLUSTERFUSION"

def _get_workspace_dir_name() -> pathlib.Path:
    try:
        with warnings.catch_warnings():
            # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
            warnings.filterwarnings(
                "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
            )
            flags = _get_cuda_arch_flags()
        arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    except Exception:
        arch = "noarch"
    return CLUSTERFUSION_CACHE_DIR / arch


# use pathlib
CLUSTERFUSION_WORKSPACE_DIR = _get_workspace_dir_name()
CLUSTERFUSION_JIT_DIR = CLUSTERFUSION_WORKSPACE_DIR / "cached_ops"
CLUSTERFUSION_GEN_SRC_DIR = CLUSTERFUSION_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[1]
CLUSTERFUSION_AOT_DIR = _package_root / "data" / "aot"
CLUSTERFUSION_INCLUDE_DIR = _package_root / "data" / "include"
CLUSTERFUSION_CSRC_DIR = _package_root / "data" / "csrc"