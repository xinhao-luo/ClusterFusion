#include "kernel.cuh"
#include <torch/extension.h>

torch::Tensor deepseek_decoder_layer(
    torch::Tensor input,
    torch::Tensor weight_q_nope,
    torch::Tensor weight_q_pe,
    torch::Tensor weight_uk,
    torch::Tensor weight_kv_nope,
    torch::Tensor weight_k_pe,
    torch::Tensor weight_uv,
    torch::Tensor weight_o,
    torch::Tensor ckv_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor rms_ckv_weight,
    torch::Tensor cos,
    torch::Tensor sin
) 
{
    cudaFuncSetAttribute(DeepSeekDecoderLayerKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({1, HIDDEN_DIM}, 0, options);
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* weight_q_nope_ptr = reinterpret_cast<half*>(weight_q_nope.data_ptr<at::Half>());
    half* weight_q_pe_ptr = reinterpret_cast<half*>(weight_q_pe.data_ptr<at::Half>());
    half* weight_uk_ptr = reinterpret_cast<half*>(weight_uk.data_ptr<at::Half>());
    half* weight_kv_nope_ptr = reinterpret_cast<half*>(weight_kv_nope.data_ptr<at::Half>());
    half* weight_k_pe_ptr = reinterpret_cast<half*>(weight_k_pe.data_ptr<at::Half>());
    half* weight_uv_ptr = reinterpret_cast<half*>(weight_uv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* ckv_cache_ptr = reinterpret_cast<half*>(ckv_cache.data_ptr<at::Half>());
    half* rms_input_weight_ptr = reinterpret_cast<half*>(rms_input_weight.data_ptr<at::Half>());
    half* rms_ckv_weight_ptr = reinterpret_cast<half*>(rms_ckv_weight.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());
    
    CUtensorMap tensor_map_weight_q{};
    CUtensorMap tensor_map_weight_q_pe{};
    CUtensorMap tensor_map_weight_uk{};
    CUtensorMap tensor_map_weight_kv{};
    CUtensorMap tensor_map_weight_k_pe{};
    CUtensorMap tensor_map_kv_cache{};
    CUtensorMap tensor_map_kv_cache_{};
    CUtensorMap tensor_map_weight_uv{};
    CUtensorMap tensor_map_weight_o{};

    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HEAD_NUM * NOPE_HEAD_DIM, HIDDEN_DIM};
    uint64_t stride[rank - 1] = {HEAD_NUM * NOPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size[rank] = {NOPE_HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride[rank] = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight_q,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_q_nope_ptr,                 
        size,                       
        stride,                     
        box_size,                   
        elem_stride,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_w_q_rope[rank] = {HEAD_NUM * ROPE_HEAD_DIM, HIDDEN_DIM};
    uint64_t stride_w_q_rope[rank - 1] = {HEAD_NUM * ROPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size_w_q_rope[rank] = {TMA_LOAD_ONCE, 128};
    uint32_t elem_stride_w_q_rope[rank] = {1, 1};
    CUresult res_2 = cuTensorMapEncodeTiled(
        &tensor_map_weight_q_pe,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_q_pe_ptr,                 
        size_w_q_rope,                       
        stride_w_q_rope,                     
        box_size_w_q_rope,                   
        elem_stride_w_q_rope,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_w_uk[rank] = {HEAD_NUM * KV_LORA_RANK, NOPE_HEAD_DIM};
    uint64_t stride_w_uk[rank - 1] = {HEAD_NUM * KV_LORA_RANK * sizeof(half)};
    uint32_t box_size_w_uk[rank] = {128, TMA_LOAD_ONCE};
    uint32_t elem_stride_w_uk[rank] = {1, 1};
    CUresult res_3 = cuTensorMapEncodeTiled(
        &tensor_map_weight_uk,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_uk_ptr,                 
        size_w_uk,                       
        stride_w_uk,                     
        box_size_w_uk,                   
        elem_stride_w_uk,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_w_kv[rank] = {KV_LORA_RANK, HIDDEN_DIM};
    uint64_t stride_w_kv[rank - 1] = {KV_LORA_RANK * sizeof(half)};
    uint32_t box_size_w_kv[rank] = {128, TMA_LOAD_ONCE};
    uint32_t elem_stride_w_kv[rank] = {1, 1};
    CUresult res_4 = cuTensorMapEncodeTiled(
        &tensor_map_weight_kv,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_kv_nope_ptr,                 
        size_w_kv,                       
        stride_w_kv,                     
        box_size_w_kv,                   
        elem_stride_w_kv,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_w_k_pe[rank] = {ROPE_HEAD_DIM, HIDDEN_DIM};
    uint64_t stride_w_k_pe[rank - 1] = {ROPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size_w_k_pe[rank] = {TMA_LOAD_ONCE, 128};
    uint32_t elem_stride_w_k_pe[rank] = {1, 1};
    CUresult res_5 = cuTensorMapEncodeTiled(
        &tensor_map_weight_k_pe,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_k_pe_ptr,                 
        size_w_k_pe,                       
        stride_w_k_pe,                     
        box_size_w_k_pe,                   
        elem_stride_w_k_pe,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_kv_cache[rank] = {MLA_HEAD_DIM, SEQ_LEN};
    uint64_t stride_kv_cache[rank - 1] = {MLA_HEAD_DIM * sizeof(half)};
    uint32_t box_size_kv_cache[rank] = {256, TMA_LOAD_ONCE_ATTN};
    uint32_t elem_stride_kv_cache[rank] = {1, 1};
    CUresult res_6 = cuTensorMapEncodeTiled(
        &tensor_map_kv_cache,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        ckv_cache_ptr,                 
        size_kv_cache,                       
        stride_kv_cache,                     
        box_size_kv_cache,                   
        elem_stride_kv_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    CUresult res_7 = cuTensorMapEncodeTiled(
        &tensor_map_kv_cache_,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        ckv_cache_ptr,                 
        size_kv_cache,                       
        stride_kv_cache,                     
        box_size_kv_cache,                   
        elem_stride_kv_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_w_uv[rank] = {HEAD_NUM * NOPE_HEAD_DIM, KV_LORA_RANK};
    uint64_t stride_w_uv[rank - 1] = {HEAD_NUM * NOPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size_w_uv[rank] = {128, TMA_LOAD_ONCE};
    uint32_t elem_stride_w_uv[rank] = {1, 1};
    CUresult res_8 = cuTensorMapEncodeTiled(
        &tensor_map_weight_uv,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_uv_ptr,                 
        size_w_uv,                       
        stride_w_uv,                     
        box_size_w_uv,                   
        elem_stride_w_uv,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_w_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_w_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_w_o[rank] = {TMA_LOAD_ONCE, NOPE_HEAD_DIM};
    uint32_t elem_stride_w_o[rank] = {1, 1};
    CUresult res_9 = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_o_ptr,                 
        size_w_o,                       
        stride_w_o,                     
        box_size_w_o,                   
        elem_stride_w_o,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    cudaDeviceSynchronize();

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    DeepSeekDecoderLayerKernel<<<grid, block>>>(
        o_ptr,
        input_ptr,
        rms_input_weight_ptr,
        rms_ckv_weight_ptr,
        cos_ptr,
        sin_ptr,
        tensor_map_weight_q,
        tensor_map_weight_q_pe,
        tensor_map_weight_uk,
        tensor_map_weight_kv,
        tensor_map_weight_k_pe,
        tensor_map_kv_cache,
        tensor_map_kv_cache_,
        tensor_map_weight_uv,
        tensor_map_weight_o
    );
    cudaDeviceSynchronize();
    return o;
}