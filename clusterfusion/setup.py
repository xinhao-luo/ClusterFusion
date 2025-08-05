from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-arch=compute_120a",
        "-code=sm_120a",
        "-lcuda",
    ],
}

# TODO: rename kernel to support both H100 and 5090
setup(
    name="clusterfusion",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="clusterfusion",
            sources=[
                "pybind.cpp",
                "kernel/5090/llama/llama_kernel_dispatch.cu",
                "kernel/H100/deepseek/deepseek_kernel_dispatch.cu",
                "kernel/H100/norm/norm_kernel_dispatch.cu",
            ],
            extra_compile_args=extra_compile_args,
            libraries=["cuda", "cudart"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)