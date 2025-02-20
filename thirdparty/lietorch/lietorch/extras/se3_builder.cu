#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define NUM_THREADS 64
// #define RADIUS 32


__global__ void se3_build_forward_kernel(
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> attention,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> transforms,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
  torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Hx,
  torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> bx,
  int radius)
{
    /* Dense transform layer aggregation step
        Inputs:
            attention:      [B, H, W, H, W]
            transforms:     [B, H, W, 4, 4]
            points:         [B, 3, H, W]
            targets:        [B, 2, H, W]
            weights:        [B, 2, H, W]
            intrinsics:     [B, 4]

        Outputs:
            Hx:             [B, H, W, 6, 6]
            bx:             [B, H, W, 6, 1]
    */
    
  int batch_id = blockIdx.x; // batch_index
  int tx = threadIdx.x;
  int ix = blockIdx.y * NUM_THREADS + tx; // image_index

  int ht = attention.size(1);
  int wd = attention.size(2);
  int dim = ht * wd;

  int h1 = ix / wd;
  int w1 = ix % wd;

  const float* Gdata = transforms[batch_id].data();
  const float* Xdata = points[batch_id].data();
  const float* rdata = targets[batch_id].data();
  const float* wdata = weights[batch_id].data();

  __shared__ float fx, fy, cx, cy;
  if (tx == 0) {
    fx = intrinsics[batch_id][0];
    fy = intrinsics[batch_id][1];
    cx = intrinsics[batch_id][2];
    cy = intrinsics[batch_id][3];
  }

  float G[12];
  if (ix < dim) {
    for (int k=0; k<12; k++)
      G[k] = Gdata[ix + k*dim];
  }

  // linear system
  float H[6][6];
  float b[6];

  for (int ii=0; ii<6; ii++) {
    b[ii] = 0.0f;
    for (int jj=0; jj<6; jj++) {
      H[ii][jj] = 0.0f;
    }
  }

  // jacobians
  float Ju[6];
  float Jv[6];
  float Jz[6];

  __shared__ float X0[3][NUM_THREADS];
  __shared__ float rvec[3][NUM_THREADS];
  __shared__ float wvec[3][NUM_THREADS];

  __syncthreads();

  for (int i=0; i<dim; i+=NUM_THREADS) {
    // load in data
    int jx = i + tx;
    if (jx < dim) {
      X0[0][tx] = Xdata[jx+0*dim];
      X0[1][tx] = Xdata[jx+1*dim];
      X0[2][tx] = Xdata[jx+2*dim];

      rvec[0][tx] = rdata[jx+0*dim];
      rvec[1][tx] = rdata[jx+1*dim];
      rvec[2][tx] = rdata[jx+2*dim];

      wvec[0][tx] = wdata[jx+0*dim];
      wvec[1][tx] = wdata[jx+1*dim];
      wvec[2][tx] = wdata[jx+2*dim];
    }

    __syncthreads();

    for (int j=0; j<NUM_THREADS; j++) {
      jx = i + j;
      if (ix<dim && jx<dim) {
        int h2 = jx / wd;
        int w2 = jx % wd;

        int r = max(abs(h1-h2), abs(w1-w2));
        if (r > radius) 
          continue;

        float w = attention[batch_id][h1][w1][h2][w2];
        float wu = w * wvec[0][j];
        float wv = w * wvec[1][j];
        float wz = w * wvec[2][j];

        float X1, Y1, Z1;
        X1 = G[0]*X0[0][j] + G[1]*X0[1][j] + G[2]*X0[2][j] + G[3];
        Y1 = G[4]*X0[0][j] + G[5]*X0[1][j] + G[6]*X0[2][j] + G[7];
        Z1 = G[8]*X0[0][j] + G[9]*X0[1][j] + G[10]*X0[2][j] + G[11];

        if (Z1 < 0.1) Z1 = 0.001;

        // residual vectors
        float ru = rvec[0][j] - (fx * (X1 / Z1) + cx);
        float rv = rvec[1][j] - (fy * (Y1 / Z1) + cy);
        float rz = rvec[2][j] - (1.0 / Z1);

        if (abs(ru) > 250 || abs(rv) > 250 || Z1 < 0.1) {
          continue;
        }

        float d = 1.f/Z1; 
        float d2 = d*d;

        // x-jacobians
        Ju[0] = fx * d;
        Ju[1] = fx * 0.0;
        Ju[2] = fx * (-X1*d2);
        Ju[3] = fx * (-X1*Y1*d2);
        Ju[4] = fx * (1 + X1*X1*d2);
        Ju[5] = fx * (-Y1*d);

        // y-jacobians
        Jv[0] = fy * 0.0;
        Jv[1] = fy * d;
        Jv[2] = fy * (-Y1*d2);
        Jv[3] = fy * -1 * (1+Y1*Y1*d2);
        Jv[4] = fy * X1*Y1*d2;
        Jv[5] = fy * X1*d;

        // z-jacobians
        Jz[0] = 0.0;
        Jz[1] = 0.0;
        Jz[2] = -d2;
        Jz[3] = d * Y1;
        Jz[4] = -d * X1;
        Jz[5] = 0.0;

        for (int ii=0; ii<6; ii++) {
          b[ii] += wu*ru*Ju[ii] + wv*rv*Jv[ii] + wz*rz*Jz[ii];
          for (int jj=0; jj<6; jj++) {
            H[ii][jj] += wu*Ju[ii]*Ju[jj] + wv*Jv[ii]*Jv[jj] + wz*Jz[ii]*Jz[jj];
          }
        }

      }
    }
    __syncthreads();
  }

  if (ix < dim) {
    for (int ii=0; ii<6; ii++) {
      bx[batch_id][ii][0][h1][w1] = b[ii];
      for (int jj=0; jj<6; jj++) {
        Hx[batch_id][ii][jj][h1][w1] = H[ii][jj];
      }
    }
  }
}


__global__ void se3_build_backward_kernel(
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> attention,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> transforms,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Hx_grad,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> bx_grad,
  torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> attention_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights_grad,
  int radius)
{

  int batch_id = blockIdx.x; // batch_index
  int tx = threadIdx.x;
  int ix = blockIdx.y * NUM_THREADS + tx; // image_index

  int ht = attention.size(1);
  int wd = attention.size(2);
  int dim = ht * wd;

  int h2 = ix / wd;
  int w2 = ix % wd;

  const float* Gdata = transforms[batch_id].data();
  const float* Hdata = Hx_grad[batch_id].data();
  const float* bdata = bx_grad[batch_id].data();

  __shared__ float fx, fy, cx, cy;
  if (tx == 0) {
    fx = intrinsics[batch_id][0];
    fy = intrinsics[batch_id][1];
    cx = intrinsics[batch_id][2];
    cy = intrinsics[batch_id][3];
  }

  float X0[3];
  X0[0] = points[batch_id][0][h2][w2];
  X0[1] = points[batch_id][1][h2][w2];
  X0[2] = points[batch_id][2][h2][w2];

  float target_u = targets[batch_id][0][h2][w2];
  float target_v = targets[batch_id][1][h2][w2];
  float target_z = targets[batch_id][2][h2][w2];

  float wu = weights[batch_id][0][h2][w2];
  float wv = weights[batch_id][1][h2][w2];
  float wz = weights[batch_id][2][h2][w2];

  // jacobians
  float Ju[6], Jv[6], Jz[6];
  float diff_ru = 0.0f;
  float diff_rv = 0.0f;
  float diff_rz = 0.0f;
  float diff_wu = 0.0f;
  float diff_wv = 0.0f;
  float diff_wz = 0.0f;

  __shared__ float Gs[12][NUM_THREADS];
  __shared__ float H_grad[36][NUM_THREADS];
  __shared__ float b_grad[6][NUM_THREADS];
  
  __syncthreads();

  for (int i=0; i<dim; i+=NUM_THREADS) {
    int jx = i + tx;
    if (jx < dim) {
      for (int k=0; k<12; k++)
        Gs[k][tx] = Gdata[jx + k*dim];

      for (int k=0; k<36; k++)
        H_grad[k][tx] = Hdata[jx + k*dim];

      for (int k=0; k<6; k++)
        b_grad[k][tx] = bdata[jx + k*dim];
    }

    __syncthreads();

    for (int j=0; j<NUM_THREADS; j++) {
      jx = i + j;
      if (ix<dim && jx<dim) {
        int h1 = jx / wd;
        int w1 = jx % wd;

        int r = max(abs(h1-h2), abs(w1-w2));
        if (r > radius) 
          continue;

        float w = attention[batch_id][h1][w1][h2][w2];
        float diff_w = 0.0f;

        float X1, Y1, Z1;
        X1 = Gs[0][j]*X0[0] + Gs[1][j]*X0[1] + Gs[2][j]*X0[2] + Gs[3][j];
        Y1 = Gs[4][j]*X0[0] + Gs[5][j]*X0[1] + Gs[6][j]*X0[2] + Gs[7][j];
        Z1 = Gs[8][j]*X0[0] + Gs[9][j]*X0[1] + Gs[10][j]*X0[2] + Gs[11][j];

        if (Z1 < 0.1) Z1 = 0.001;

        // residual vectors
        float ru = target_u - (fx * (X1 / Z1) + cx);
        float rv = target_v - (fy * (Y1 / Z1) + cy);
        float rz = target_z - (1.0 / Z1);

        if (abs(ru) > 50 || abs(rv) > 50 || Z1 < 0.1) {
          continue;
        }

        float d = 1.f/Z1; 
        float d2 = d*d;

        // x-jacobians
        Ju[0] = fx * d;
        Ju[1] = fx * 0.0;
        Ju[2] = fx * (-X1*d2);
        Ju[3] = fx * (-X1*Y1*d2);
        Ju[4] = fx * (1 + X1*X1*d2);
        Ju[5] = fx * (-Y1*d);

        // y-jacobians
        Jv[0] = fy * 0.0;
        Jv[1] = fy * d;
        Jv[2] = fy * (-Y1*d2);
        Jv[3] = fy * -1 * (1+Y1*Y1*d2);
        Jv[4] = fy * X1*Y1*d2;
        Jv[5] = fy * X1*d;

        // z-jacobians
        Jz[0] = 0.0;
        Jz[1] = 0.0;
        Jz[2] = -d2;
        Jz[3] = d * Y1;
        Jz[4] = -d * X1;
        Jz[5] = 0.0;

        for (int ii=0; ii<6; ii++) {
          // residual gradients
          diff_ru += w*wu*Ju[ii]*b_grad[ii][j];
          diff_rv += w*wv*Jv[ii]*b_grad[ii][j];
          diff_rz += w*wz*Jz[ii]*b_grad[ii][j];

          // weights gradients
          diff_wu += w*ru*Ju[ii]*b_grad[ii][j];
          diff_wv += w*rv*Jv[ii]*b_grad[ii][j];
          diff_wz += w*rz*Jz[ii]*b_grad[ii][j];

          // embedding weight
          diff_w += (wu*ru*Ju[ii] + wv*rv*Jv[ii] + wz*rz*Jz[ii]) * b_grad[ii][j];

          for (int jj=0; jj<6; jj++) {
            diff_wu += w*Ju[ii]*Ju[jj]*H_grad[6*ii+jj][j];
            diff_wv += w*Jv[ii]*Jv[jj]*H_grad[6*ii+jj][j];
            diff_wz += w*Jz[ii]*Jz[jj]*H_grad[6*ii+jj][j];
            diff_w += (wu*Ju[ii]*Ju[jj] + wv*Jv[ii]*Jv[jj] + wz*Jz[ii]*Jz[jj])*H_grad[6*ii+jj][j];
          }
        }

        attention_grad[batch_id][h1][w1][h2][w2] = diff_w;
      }
    }
    __syncthreads();
  }

  targets_grad[batch_id][0][h2][w2] = diff_ru;
  targets_grad[batch_id][1][h2][w2] = diff_rv;
  targets_grad[batch_id][2][h2][w2] = diff_rz;

  weights_grad[batch_id][0][h2][w2] = diff_wu;
  weights_grad[batch_id][1][h2][w2] = diff_wv;
  weights_grad[batch_id][2][h2][w2] = diff_wz;
}


std::vector<torch::Tensor> se3_build_cuda(
  torch::Tensor attention,
  torch::Tensor transforms,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  int radius)
{

  int batch_size = attention.size(0);
  int ht = attention.size(1);
  int wd = attention.size(2);

  dim3 grid = dim3(batch_size, (ht*wd + NUM_THREADS-1) / NUM_THREADS);

  auto opts = attention.options();
  torch::Tensor H = torch::zeros({batch_size, 6, 6, ht, wd}, opts);
  torch::Tensor b = torch::zeros({batch_size, 6, 1, ht, wd}, opts);

  se3_build_forward_kernel<<<grid, NUM_THREADS>>>(
    attention.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    transforms.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    H.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

  return {H, b};
}


std::vector<torch::Tensor> se3_build_backward_cuda(
  torch::Tensor attention,
  torch::Tensor transforms,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  torch::Tensor H_grad,
  torch::Tensor b_grad,
  int radius)
{

  int batch_size = attention.size(0);
  int ht = attention.size(1);
  int wd = attention.size(2);

  dim3 grid = dim3(batch_size, (ht*wd + NUM_THREADS-1) / NUM_THREADS);
  torch::Tensor attention_grad = torch::zeros_like(attention);
  torch::Tensor targets_grad = torch::zeros_like(targets);
  torch::Tensor weights_grad = torch::zeros_like(weights);

  se3_build_backward_kernel<<<grid, NUM_THREADS>>>(
    attention.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    transforms.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    H_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    attention_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    targets_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    radius);

  return {attention_grad, targets_grad, weights_grad};
}