## Benchmarks
For benchmarking, please clone the following frameworks and set up their environments according to their official documentation (links provided below).


#### SGLang [Doc](https://docs.sglang.ai/start/install.html)
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model .../llama-2-7b --batch-size 16 --input-len 512 --output-len 512

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model ../DeepSeek-V2-Lite-Chat --trust-remote-code --enable-flashinfer-mla --batch-size 16 --input-len 1024 --output-len 8
```

#### vLLM [Doc](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmark_latency.py --model .../llama-2-7b --batch-size 1 --input-len 1024 --output-len 8

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmark_latency.py --model /cephfs/shared/model/DeepSeek-V2-Lite-Chat --trust-remote-code --max-model-len=32768 --batch-size 16 --input-len 1024 --output-len 8
```

#### TRT-LLM [Doc](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmarks/python/benchmark.py -m dec --batch_size 16 --input_output_len 64,64 --num_runs 1 --engine_dir .../llama-2-7b_trt_engine

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmarks/python/benchmark.py -m dec --batch_size 16 --input_output_len 64,64 --num_runs 1 --dtype bfloat16 --engine_dir .../DeepSeek-V2-Lite-Chat_trt_engine
```

#### MLC-LLM [Doc](https://llm.mlc.ai/docs/install/mlc_llm.html)
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -t cuda python tests/python/serve/evaluate_engine.py --batch-size 16 --model-lib .../llama-2-7b-q0f16-MLC/llama-2-7b-q0f16-MLC-cuda.so

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python tests/python/serve/evaluate_engine.py --batch-size 16 --model-lib .../DeepSeek-V2-Lite-Chat-MLC/DeepSeek-V2-Lite-Chat-MLC-cuda.so
```

