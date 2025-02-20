
#ifndef LIETORCH_GPU_H_
#define LIETORCH_GPU_H_

#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// unary operations
torch::Tensor exp_forward_gpu(int, torch::Tensor);
std::vector<torch::Tensor> exp_backward_gpu(int, torch::Tensor, torch::Tensor);

torch::Tensor log_forward_gpu(int, torch::Tensor);
std::vector<torch::Tensor> log_backward_gpu(int, torch::Tensor, torch::Tensor);

torch::Tensor inv_forward_gpu(int, torch::Tensor);
std::vector<torch::Tensor> inv_backward_gpu(int, torch::Tensor, torch::Tensor);

// binary operations
torch::Tensor mul_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> mul_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor adj_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> adj_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor adjT_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> adjT_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor act_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> act_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor act4_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> act4_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

// conversion operations
// std::vector<torch::Tensor> to_vec_backward_gpu(int, torch::Tensor, torch::Tensor);
// std::vector<torch::Tensor> from_vec_backward_gpu(int, torch::Tensor, torch::Tensor);

// utility operators
torch::Tensor orthogonal_projector_gpu(int, torch::Tensor);

torch::Tensor as_matrix_forward_gpu(int, torch::Tensor);

torch::Tensor jleft_forward_gpu(int, torch::Tensor, torch::Tensor);

#endif


  