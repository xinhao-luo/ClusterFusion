#include <iostream>
#include "decode_kernel.cuh"

// CUDA_VISIBLE_DEVICES=0 nvcc --generate-code=arch=compute_120a,code=sm_120a -O3 -std=c++17 -lcuda decode.cu -o test && ./test

int main(int argc, char** argv) {
    cudaFuncSetAttribute(LlamaDecoderLayerKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    uint32_t max_shmem_size = ((((DIM_PER_BLOCK * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float) + 127) & ~127) +  2 * TMA_LOAD_ONCE * MAX_SMEM_DIM * sizeof(half) + 127) & ~127) + (3 * HEAD_DIM) * sizeof(half);
    cudaFuncSetAttribute(LlamaDecoderLayerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    // Init input
    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_qkv, *d_w_qkv;
    half *h_w_o, *d_w_o;
    half *h_ffn_gate_up, *d_ffn_gate_up;
    half *h_ffn_down, *d_ffn_down;
    half *h_rms_input, *d_rms_input;
    half *h_rms_attn, *d_rms_attn;
    float *h_cos, *d_cos;
    float *h_sin, *d_sin;
    h_input = new half[1 * HIDDEN_DIM];
    h_w_qkv = new half[3 * HIDDEN_DIM * HIDDEN_DIM];
    h_w_o = new half[HIDDEN_DIM * HIDDEN_DIM];
    h_k_cache = new half[SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_v_cache = new half[SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_ffn_gate_up = new half[2 * HIDDEN_DIM * FFN_DIM];
    h_ffn_down = new half[FFN_DIM * HIDDEN_DIM];
    h_rms_input = new half[HIDDEN_DIM];
    h_rms_attn = new half[HIDDEN_DIM];
    h_cos = new float[HEAD_DIM];
    h_sin = new float[HEAD_DIM];

    fill_matrix(h_input, 1 * HIDDEN_DIM);
    fill_matrix(h_w_qkv, 3 * HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_w_o, HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_k_cache, SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_v_cache, SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_ffn_gate_up, 2 * HIDDEN_DIM * FFN_DIM);
    fill_matrix(h_ffn_down, FFN_DIM * HIDDEN_DIM);
    fill_matrix(h_rms_input, HIDDEN_DIM);
    fill_matrix(h_rms_attn, HIDDEN_DIM);

    // Init cos, sin used in RoPE
    int encode_point_offset = 0;
    float rope_scale = 1;
    float rope_theta = 500000;
    for (int j = 0; j < HEAD_DIM; j++) {
        float inv_freq =(encode_point_offset / rope_scale) / (std::pow(rope_theta, float(2 * (j % (HEAD_DIM / 2))) / float(HEAD_DIM)));
        h_cos[j] = std::cos(inv_freq);
        h_sin[j] = std::sin(inv_freq);
    }

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 1 * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_qkv), sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_gate_up), sizeof(half) * 2 * HIDDEN_DIM * FFN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_down), sizeof(half) * FFN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_attn), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_cos), sizeof(float) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_sin), sizeof(float) * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_qkv), h_w_qkv, sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_gate_up), h_ffn_gate_up, sizeof(half) * 2 * HIDDEN_DIM * FFN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_down), h_ffn_down, sizeof(half) * FFN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_input), h_rms_input, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_attn), h_rms_attn, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_cos), h_cos, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_sin), h_sin, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[1 * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * 1 * HIDDEN_DIM);
    
    half *global_reduce;
    cudaMalloc(reinterpret_cast<void**>(&global_reduce), sizeof(half) * HIDDEN_DIM);
    
    CUtensorMap tensor_map_weight{};
    CUtensorMap tensor_map_k_cache{};
    CUtensorMap tensor_map_v_cache{};
    CUtensorMap tensor_map_weight_o{};
    // CUtensorMap tensor_map_weight_gate_up{};
    // CUtensorMap tensor_map_weight_gate_up_{};
    // CUtensorMap tensor_map_weight_down{};
    // CUtensorMap tensor_map_weight_down_{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HIDDEN_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride[rank] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_w_qkv,                
        size,                      
        stride,                     
        box_size,                   
        elem_stride,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};

    CUresult res_k_cache = cuTensorMapEncodeTiled(
        &tensor_map_k_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_k_cache,                
        size_k_cache,                      
        stride_k_cache,                     
        box_size_k_cache,                   
        elem_stride_k_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_v_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_v_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_v_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_v_cache[rank] = {1, 1};

    CUresult res_v_cache = cuTensorMapEncodeTiled(
        &tensor_map_v_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_v_cache,                
        size_v_cache,                      
        stride_v_cache,                     
        box_size_v_cache,                   
        elem_stride_v_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_weight_o[rank] = {1, 1};

    CUresult res_weight_o = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_w_o,                
        size_weight_o,                      
        stride_weight_o,                     
        box_size_weight_o,                   
        elem_stride_weight_o,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // uint64_t size_weight_gate_up[rank] = {FFN_DIM, 2 * HIDDEN_DIM};
    // uint64_t stride_weight_gate_up[rank - 1] = {FFN_DIM * sizeof(half)};
    // uint32_t box_size_weight_gate_up[rank] = {TMA_LOAD_ONCE_MAX, TMA_LOAD_ONCE};
    // uint32_t elem_stride_weight_gate_up[rank] = {1, 1};

    // CUresult res_weight_gate_up = cuTensorMapEncodeTiled(
    //     &tensor_map_weight_gate_up,               
    //     CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    //     rank,                      
    //     d_ffn_gate_up,                
    //     size_weight_gate_up,                      
    //     stride_weight_gate_up,                     
    //     box_size_weight_gate_up,                   
    //     elem_stride_weight_gate_up,                
    //     CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    //     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    //     CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    //     CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    // );

    // uint64_t size_weight_gate_up_[rank] = {FFN_DIM, 2 * HIDDEN_DIM};
    // uint64_t stride_weight_gate_up_[rank - 1] = {FFN_DIM * sizeof(half)};
    // uint32_t box_size_weight_gate_up_[rank] = {FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX, TMA_LOAD_ONCE};
    // uint32_t elem_stride_weight_gate_up_[rank] = {1, 1};

    // CUresult res_weight_gate_up_ = cuTensorMapEncodeTiled(
    //     &tensor_map_weight_gate_up_,               
    //     CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    //     rank,                      
    //     d_ffn_gate_up,                
    //     size_weight_gate_up_,                      
    //     stride_weight_gate_up_,                     
    //     box_size_weight_gate_up_,                   
    //     elem_stride_weight_gate_up_,                
    //     CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    //     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    //     CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    //     CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    // );

    // uint64_t size_weight_down[rank] = {HIDDEN_DIM, FFN_DIM};
    // uint64_t stride_weight_down[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    // uint32_t box_size_weight_down[rank] = {TMA_LOAD_ONCE, TMA_LOAD_ONCE_MAX};
    // uint32_t elem_stride_weight_down[rank] = {1, 1};

    // CUresult res_weight_down = cuTensorMapEncodeTiled(
    //     &tensor_map_weight_down,               
    //     CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    //     rank,                      
    //     d_ffn_down,                
    //     size_weight_down,                      
    //     stride_weight_down,                     
    //     box_size_weight_down,                   
    //     elem_stride_weight_down,                
    //     CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    //     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    //     CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    //     CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    // );

    // uint64_t size_weight_down_[rank] = {HIDDEN_DIM, FFN_DIM};
    // uint64_t stride_weight_down_[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    // uint32_t box_size_weight_down_[rank] = {TMA_LOAD_ONCE, FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX};
    // uint32_t elem_stride_weight_down_[rank] = {1, 1};

    // CUresult res_weight_down_ = cuTensorMapEncodeTiled(
    //     &tensor_map_weight_down_,               
    //     CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    //     rank,                      
    //     d_ffn_down,                
    //     size_weight_down_,                      
    //     stride_weight_down_,                     
    //     box_size_weight_down_,                   
    //     elem_stride_weight_down_,                
    //     CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    //     CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    //     CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    //     CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    // );

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 100;
    int test = 100;
    for (int i = 0; i < wmup; i++) {
        LlamaDecoderLayerKernel<<<grid, block, max_shmem_size>>>(
            d_output,
            d_input,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin,
            tensor_map_weight,
            tensor_map_k_cache,
            tensor_map_v_cache,
            tensor_map_weight_o
        );
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        LlamaDecoderLayerKernel<<<grid, block, max_shmem_size>>>(
            d_output,
            d_input,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin,
            tensor_map_weight,
            tensor_map_k_cache,
            tensor_map_v_cache,
            tensor_map_weight_o
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < HIDDEN_DIM; i++)
    //     printf("%f, ", __half2float(h_output[i]));
    // printf("\n");
    return 0;
}