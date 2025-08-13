<!-- # ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive

This repository contains the official implementation of **ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive**.

## Requirements

- CUDA 12.4  
- PyTorch 2.5.1  
- Python 3.12 
- Flashinfer 0.2.0
- SGLang 0.4.3.post2
- vLLM 0.6.4.post1
- TensorRT-LLM 0.18.0
- MLC-LLM 0.20.dev0

## Installation

To install **ClusterFusion**, navigate to the `/clusterfusion` directory and run:

```bash
cd clusterfusion
python setup.py install
```

## Tests
To verify the correctness of **ClusterFusion**, run the following test scripts:
```bash
cd tests
python test_llama.py
python test_deepseek.py
```

## Evaluation

Benchmark scripts are provided in the `/benchmarks/E2E.md`.  
NVIDIA Nsight Systems (Nsys) reports can be downloaded from the following link: [Download Nsys Report](https://drive.google.com/file/d/1t8uuTfv0VFeVX3ICoS9xtqh1LoaOjk1S/view?usp=sharing).

The complete codebase, along with evaluation scripts located at the top of each code file, is available in the `/workspace` directory. For example, you can test the performance of the core modules with the following command:
```bash
cd workspace
CUDA_VISIBLE_DEVICES=0 nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda linear-attn.cu -o test && ./test
```
Configurations such as batch size and sequence length can be modified in `config.h` or `config_linear-attn.h`.
 -->
