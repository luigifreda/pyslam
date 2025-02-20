#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#define BLOCK 16

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void corr_index_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr,
    int r)
{
  // batch index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume.size(1);
  const int w1 = volume.size(2);
  const int h2 = volume.size(3);
  const int w2 = volume.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  float x0 = coords[n][0][y][x];
  float y0 = coords[n][1][y][x];

  float dx = x0 - floor(x0);
  float dy = y0 - floor(y0);

  int rd = 2*r + 1;
  for (int i=0; i<rd+1; i++) {
    for (int j=0; j<rd+1; j++) {
      int x1 = static_cast<int>(floor(x0)) - r + i;
      int y1 = static_cast<int>(floor(y0)) - r + j;

      if (within_bounds(y1, x1, h2, w2)) {
        scalar_t s = volume[n][y][x][y1][x1];

        if (i > 0 && j > 0)
          corr[n][i-1][j-1][y][x] += s * scalar_t(dx * dy);

        if (i > 0 && j < rd)
          corr[n][i-1][j][y][x] += s * scalar_t(dx * (1.0f-dy));

        if (i < rd && j > 0)
          corr[n][i][j-1][y][x] += s * scalar_t((1.0f-dx) * dy);

        if (i < rd && j < rd)
          corr[n][i][j][y][x] += s * scalar_t((1.0f-dx) * (1.0f-dy));

      }
    }
  }
}


template <typename scalar_t>
__global__ void corr_index_backward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume_grad,
    int r)
{
  // batch index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume_grad.size(1);
  const int w1 = volume_grad.size(2);
  const int h2 = volume_grad.size(3);
  const int w2 = volume_grad.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  float x0 = coords[n][0][y][x];
  float y0 = coords[n][1][y][x];

  float dx = x0 - floor(x0);
  float dy = y0 - floor(y0);

  int rd = 2*r + 1;
  for (int i=0; i<rd+1; i++) {
    for (int j=0; j<rd+1; j++) {
      int x1 = static_cast<int>(floor(x0)) - r + i;
      int y1 = static_cast<int>(floor(y0)) - r + j;

      if (within_bounds(y1, x1, h2, w2)) {
        scalar_t g = 0.0;
        if (i > 0 && j > 0)
          g += corr_grad[n][i-1][j-1][y][x] * scalar_t(dx * dy);

        if (i > 0 && j < rd)
          g += corr_grad[n][i-1][j][y][x] * scalar_t(dx * (1.0f-dy));

        if (i < rd && j > 0)
          g += corr_grad[n][i][j-1][y][x] * scalar_t((1.0f-dx) * dy);

        if (i < rd && j < rd)
          g += corr_grad[n][i][j][y][x] * scalar_t((1.0f-dx) * (1.0f-dy));

        volume_grad[n][y][x][y1][x1] += g;
      }
    }
  }
}

std::vector<torch::Tensor> corr_index_cuda_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK, BLOCK);

  auto opts = volume.options();
  torch::Tensor corr = torch::zeros(
    {batch_size, 2*radius+1, 2*radius+1, ht, wd}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
    corr_index_forward_kernel<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      corr.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {corr};

}

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  auto volume_grad = torch::zeros_like(volume);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel", ([&] {
    corr_index_backward_kernel<scalar_t><<<blocks, threads>>>(
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      volume_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {volume_grad};
}






