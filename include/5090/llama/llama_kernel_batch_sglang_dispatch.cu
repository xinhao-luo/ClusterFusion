#include "kernel_batch_sglang.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>


void llama_decoder_layer_batch_sglang_sm120(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
)
{
    cudaFuncSetAttribute(LlamaDecoderLayerBatchDecodeWithPagedKVCacheKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    // uint32_t max_shmem_size = ((((DIM_PER_BLOCK * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float) + 127) & ~127) +  2 * TMA_LOAD_ONCE * MAX_SMEM_DIM * sizeof(half) + 127) & ~127) + (3 * HEAD_DIM) * sizeof(half);
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(LlamaDecoderLayerBatchDecodeWithPagedKVCacheKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    uint32_t batch_size = input.size(0);
    half* o_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    half* residual_output_ptr = reinterpret_cast<half*>(residual_output.data_ptr<at::Half>());

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* residual_ptr = reinterpret_cast<half*>(residual.data_ptr<at::Half>());
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    uint64_t* k_cache_ptrs_array = reinterpret_cast<uint64_t*>(k_cache_ptrs.data_ptr<uint64_t>());
    uint64_t* v_cache_ptrs_array = reinterpret_cast<uint64_t*>(v_cache_ptrs.data_ptr<uint64_t>());
    half* rms_input_weight_ptr = reinterpret_cast<half*>(rms_input_weight.data_ptr<at::Half>());
    int64_t* positions_ptr = reinterpret_cast<int64_t*>(positions.data_ptr<int64_t>());
    float* cos_sin_ptr = reinterpret_cast<float*>(cos_sin.data_ptr<float>());
    int* paged_kv_indptr_ptr = reinterpret_cast<int*>(paged_kv_indptr.data_ptr<int>());
    int* paged_kv_indices_ptr = reinterpret_cast<int*>(paged_kv_indices.data_ptr<int>());
    
    CUtensorMap tensor_map_weight{};
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

    dim3 grid(HEAD_NUM * CLUSTER_SIZE * batch_size); 
    dim3 block(BLOCK_SIZE);

    LlamaDecoderLayerBatchDecodeWithPagedKVCacheKernel<<<grid, block, max_shmem_size, stream>>>(
        o_ptr,
        input_ptr,
        residual_ptr,
        residual_output_ptr,
        rms_input_weight_ptr,
        eps,
        positions_ptr,
        cos_sin_ptr,
        k_cache_ptrs_array,
        v_cache_ptrs_array,
        layer_id,
        paged_kv_indptr_ptr,
        paged_kv_indices_ptr,
        tensor_map_weight,
        tensor_map_weight_o
    );

    return;
}