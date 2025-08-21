run:
	CUDA_VISIBLE_DEVICES=0 nvcc --generate-code=arch=compute_120a,code=sm_120a -O3 -std=c++17 -lcuda -I/sgl-workspace/clusterfusion/cutlass/include decode.cu -o test && ./test && rm test