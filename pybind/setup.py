from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

A100_extra_compile_args = {
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-arch=sm_80",
        "-lcuda",
        "-Xptxas=-v", 
        "-Xptxas=-warn-lmem-usage"
    ],
}

setup(
    name="fuse_all",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fuse_all",
            sources=[
                "pybind.cpp",
                "kernel/dummy_single_decode_layer.cu",
            ],
            extra_compile_args=A100_extra_compile_args,
            libraries=["cuda", "cudart"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)