#include <torch/extension.h>
#include <vector>


#define NUM_THREADS 64
#define EPS 1e-8


template <int N>
__device__ __forceinline__ void llt(const float A[N][N], float L[N][N])
{

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      L[i][j] = 0;
    }
  }

  float s;
  for (int i=0; i<N; i++) {
    for (int j=0; j<(i+1); j++) {
      s = 0.0;
      for (int k=0; k<j; k++)
        s += L[i][k] * L[j][k];

      if (i==j) {
        s = s > A[i][i] ? A[i][i] + EPS : s;
        L[i][j] = sqrtf(A[i][i]-s);
      }

      else
        L[i][j] = (A[i][j] - s) / L[j][j];
    }
  }
}

template <int N> 
__device__ __forceinline__ void llt_solve(const float L[N][N], float x[N])
{
  float s;
  for (int i=0; i<N; i++) {
    s = 0.0;
    for (int j=0; j<i; j++)
      s += L[i][j] * x[j];

    x[i] = (x[i] - s) / L[i][i];
  }

  for (int i=N-1; i>=0; i--) {
    s = 0.0;
    for (int j=i+1; j<N; j++)
      s += L[j][i] * x[j];

    x[i] = (x[i] - s) / L[i][i];
  }
}


__global__ void cholesky_solve6x6_forward_kernel(
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> H_tensor,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> b_tensor,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> x_tensor) {

  /*Inputs: H [batch,6,6,ht,wd], b [batch,6,1,ht,wd]
    Outputs: x [batch,6,1,ht,wd]; Hx = b
  */

  int batch_id = blockIdx.x;
  const int dim = H_tensor.size(3) * H_tensor.size(4);
  int m = blockIdx.y * NUM_THREADS + threadIdx.x;

  const float* H_ptr = H_tensor[batch_id].data();
  const float* b_ptr = b_tensor[batch_id].data();
  float* x_ptr = x_tensor[batch_id].data();

  if (m < dim) {
    float H[6][6], L[6][6], x[6];

    for (int i=0; i<6; i++) {
      for (int j=0; j<6; j++) {
        H[i][j] = H_ptr[m + (6*i+j)*dim];
      }
    }

    for (int i=0; i<6; i++) {
      x[i] = b_ptr[m + i*dim];
    }

    llt<6>(H, L);
    llt_solve<6>(L, x);

    for (int i=0; i<6; i++) {
      x_ptr[m + i*dim] = x[i];
    }
  }
}
  

__global__ void cholesky_solve6x6_backward_kernel(
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> H_tensor,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> b_tensor,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> dx_tensor,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> dH_tensor,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> db_tensor) {


  int batch_id = blockIdx.x;
  const int dim = H_tensor.size(3) * H_tensor.size(4);
  int m = blockIdx.y * NUM_THREADS + threadIdx.x;

  const float* H_ptr = H_tensor[batch_id].data();
  const float* b_ptr = b_tensor[batch_id].data();
  
  const float* dx_ptr = dx_tensor[batch_id].data();
  float* dH_ptr = dH_tensor[batch_id].data();
  float* db_ptr = db_tensor[batch_id].data();

  if (m < dim) {
    float H[6][6], L[6][6], x[6], dz[6];

    for (int i=0; i<6; i++) {
      for (int j=0; j<6; j++) {
        H[i][j] = H_ptr[m + (6*i+j)*dim];
      }
    }

    for (int i=0; i<6; i++) {
      x[i] = b_ptr[m + i*dim];
    }

    for (int i=0; i<6; i++) {
      dz[i] = dx_ptr[m + i*dim];
    }

    // cholesky factorization
    llt<6>(H, L);

    llt_solve<6>(L, x);
    llt_solve<6>(L, dz);

    for (int i=0; i<6; i++) {
      for (int j=0; j<6; j++) {
        dH_ptr[m + (6*i+j)*dim] = -dz[i] * x[j];
      }
    }

    for (int i=0; i<6; i++) {
      db_ptr[m + i*dim] = dz[i];
    }
  }
}


std::vector<torch::Tensor> cholesky_solve6x6_forward_cuda(torch::Tensor H, torch::Tensor b) {

  const int batch_size = H.size(0);
  const int ht = H.size(3);
  const int wd = H.size(4);

  torch::Tensor x = torch::zeros_like(b);
  dim3 grid = dim3(batch_size, (ht*wd + NUM_THREADS-1) / NUM_THREADS);

  cholesky_solve6x6_forward_kernel<<<grid, NUM_THREADS>>>(
    H.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    x.packed_accessor32<float,5,torch::RestrictPtrTraits>());

  return {x};
}


std::vector<torch::Tensor> cholesky_solve6x6_backward_cuda(torch::Tensor H, torch::Tensor b, torch::Tensor dx) {
  const int batch_size = H.size(0);
  const int ht = H.size(3);
  const int wd = H.size(4);

  torch::Tensor dH = torch::zeros_like(H);
  torch::Tensor db = torch::zeros_like(b);
  dim3 grid = dim3(batch_size, (ht*wd + NUM_THREADS-1) / NUM_THREADS);

  cholesky_solve6x6_backward_kernel<<<grid, NUM_THREADS>>>(
    H.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    dH.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    db.packed_accessor32<float,5,torch::RestrictPtrTraits>());

  return {dH, db};
}