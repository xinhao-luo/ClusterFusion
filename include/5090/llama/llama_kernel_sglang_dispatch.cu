#include "kernel_sglang.cuh"
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm120(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
) 
{
    cudaFuncSetAttribute(LlamaDecoderLayerKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    // uint32_t max_shmem_size = ((((DIM_PER_BLOCK * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float) + 127) & ~127) +  2 * TMA_LOAD_ONCE * MAX_SMEM_DIM * sizeof(half) + 127) & ~127) + (3 * HEAD_DIM) * sizeof(half);
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(LlamaDecoderLayerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({1, HIDDEN_DIM}, 0, options);
    torch::Tensor k = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    torch::Tensor v = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* residual_ptr = reinterpret_cast<half*>(residual.data_ptr<at::Half>());
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());
    half* rms_input_weight_ptr = reinterpret_cast<half*>(rms_input_weight.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());

    const uint32_t SEQ_LEN = k_cache.size(0);
    const uint32_t KV_DIM_PER_BLOCK = ((SEQ_LEN + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);
    
    CUtensorMap tensor_map_weight{};
    CUtensorMap tensor_map_k_cache{};
    CUtensorMap tensor_map_v_cache{};
    CUtensorMap tensor_map_weight_o{};
    
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HIDDEN_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride[rank] = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_qkv_ptr,                 
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
        k_cache_ptr,                
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
        v_cache_ptr,                
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
    uint32_t box_size_weight_o[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_o[rank] = {1, 1};
    CUresult res_weight_o = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_o_ptr,                 
        size_weight_o,                       
        stride_weight_o,                     
        box_size_weight_o,                   
        elem_stride_weight_o,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    LlamaDecoderLayerKernel<<<grid, block, max_shmem_size>>>(
        o_ptr,
        k_ptr,
        v_ptr,
        input_ptr,
        residual_ptr,
        rms_input_weight_ptr,
        eps,
        cos_ptr,
        sin_ptr,
        k_cache_ptr,
        v_cache_ptr,
        tensor_map_weight,
        tensor_map_k_cache,
        tensor_map_v_cache,
        tensor_map_weight_o,
        SEQ_LEN,
        KV_DIM_PER_BLOCK
    );
    cudaDeviceSynchronize();
    return std::make_tuple(o, residual, k, v);
}