#### SGLang
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model cephfs/shared/model/llama-7b-hf-transformers-4.29 --batch-size 1 --input-len 512 --output-len 512

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model /cephfs/shared/model/DeepSeek-V2-Lite-Chat --trust-remote-code --enable-flashinfer-mla --batch-size 1 --input-len 1024 --output-len 8
```

#### vLLM
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmark_latency.py --model /cephfs/shared/model/llama-7b-hf-transformers-4.29 --batch-size 1 --input-len 1024 --output-len 8

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmark_latency.py --model /cephfs/shared/model/DeepSeek-V2-Lite-Chat --trust-remote-code --max-model-len=32768 --batch-size 1 --input-len 1024 --output-len 8
```

#### TRT-LLM
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmarks/python/benchmark.py -m dec --batch_size 1 --input_output_len 64,64 --num_runs 1 --engine_dir /cephfs/shared/model/llama-7b-hf-transformers-4.29_trt_engine

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python benchmarks/python/benchmark.py -m dec --batch_size 1 --input_output_len 64,64 --num_runs 1 --dtype bfloat16 --engine_dir /cephfs/shared/model/DeepSeek-V2-Lite-Chat_trt_engine
```

#### MLC-LLM
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python tests/python/serve/evaluate_engine.py --batch-size 1 --model-lib /cephfs/shared/model/llama-7b-hf-transformers-4.29-q0f16-MLC/llama-7b-hf-transformers-4.29-q0f16-MLC-cuda.so

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python tests/python/serve/evaluate_engine.py --batch-size 1 --model-lib /cephfs/shared/model/DeepSeek-V2-Lite-Chat-MLC/DeepSeek-V2-Lite-Chat-MLC-cuda.so
```