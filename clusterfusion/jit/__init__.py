import ctypes
import functools
import os

# Re-export
from . import env as env
from .attention import cudnn_fmha_gen_module as cudnn_fmha_gen_module
from .attention import gen_customize_batch_decode_module as gen_customize_batch_decode_module
from .attention import gen_customize_batch_prefill_module as gen_customize_batch_prefill_module
from .attention import get_batch_attention_uri as get_batch_attention_uri
from .cubin_loader import setup_cubin_loader


@functools.cache
def get_cudnn_fmha_gen_module():
    mod = cudnn_fmha_gen_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


cuda_lib_path = os.environ.get(
    "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
)
if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
    ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)

