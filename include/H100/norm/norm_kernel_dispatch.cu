#include "kernel.cuh"
#include <torch/extension.h>

torch::Tensor rmsnorm(
    torch::Tensor input,
    torch::Tensor weight
) 
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({BATCH_SIZE, HIDDEN_DIM}, 0, options);
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* weight_ptr = reinterpret_cast<half*>(weight.data_ptr<at::Half>());
    
    dim3 grid(BATCH_SIZE * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    RMSNormKernel<<<grid, block>>>(
        o_ptr,
        input_ptr,
        weight_ptr
    );
    cudaDeviceSynchronize();
    return o;
}