from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

def _get_arch():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    sm = props.major * 10 + props.minor
    if sm == 90:
        return "sm90a", "COMPILE_SM90"
    elif sm == 120:
        return "sm120a", "COMPILE_SM120"
    else:
        raise RuntimeError(f"Unsupported SM version: {sm}")

_arch, _macro = _get_arch()

if _arch == "sm120a":
    sources = [
        "include/pybind.cpp",
        "include/5090/llama/llama_kernel_dispatch.cu",
    ]
    gencode = "-gencode=arch=compute_120a,code=sm_120a"
elif _arch == "sm90a":
    sources = [
        "include/pybind.cpp",
        "include/H100/llama/llama_kernel_dispatch.cu",
        "include/H100/deepseek/deepseek_kernel_dispatch.cu",
        "include/H100/norm/norm_kernel_dispatch.cu",
    ]
    gencode = "-gencode=arch=compute_90a,code=sm_90a"
else:
    raise RuntimeError(f"Unsupported arch: {_arch}")

module_name = "_clusterfusion"

setup(
    name="clusterfusion",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="clusterfusion._clusterfusion",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", f"-D{_macro}"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    gencode,
                    "-lcuda",
                    f"-D{_macro}",
                ],
            },
            libraries=["cuda", "cudart"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
