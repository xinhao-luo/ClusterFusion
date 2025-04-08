#### SGLang
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model cephfs/shared/model/llama-7b-hf-transformers-4.29 --batch-size 1 --input-len 512 --output-len 512
```

#### TRT-LLM
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --trace=cuda python benchmarks/python/benchmark.py -m dec --batch_size 1 --input_output_len 64,64 --num_runs 1 --engine_dir /cephfs/shared/model/llama-7b-hf-transformers-4.29_trt_engine
```

#### Torch/Inductor
```
python run_e2e.py -c torch/dynamo
```

#### MLC-LLM
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python tests/python/serve/evaluate_engine.py --batch-size 1 --model-lib /cephfs/shared/model/llama-7b-hf-transformers-4.29-q0f16-MLC/llama-7b-hf-transformers-4.29-q0f16-MLC-cuda.so
```

#### vLLM
```

```