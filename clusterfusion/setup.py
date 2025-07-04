from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-arch=compute_90a",
        "-code=sm_90a",
        "-lcuda",
    ],
}

setup(
    name="clusterfusion",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="clusterfusion",
            sources=[
                "pybind.cpp",
                "kernel/llama/llama_kernel_dispatch.cu",
                "kernel/deepseek/deepseek_kernel_dispatch.cu",
                "kernel/Norm/norm_kernel_dispatch.cu",
            ],
            extra_compile_args=extra_compile_args,
            libraries=["cuda", "cudart"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)