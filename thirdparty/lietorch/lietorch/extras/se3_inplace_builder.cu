#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define NUM_THREADS 64
#define AE_DIM 32

__device__ __forceinline__ float sigmoid(float x) {
  return exp(x) / (exp(x) + 1.0);
}

__device__ __forceinline__ void
se3_transform_point_inplace(const float T[7], float X[3]) {
  const float tx=T[0], ty=T[1], tz=T[2];
  const float qx=T[3], qy=T[4], qz=T[5], qw=T[6];  
  
  float uv[3];
  uv[0] = 2.0 * (qy*X[2] - qz*X[1]);
  uv[1] = 2.0 * (qz*X[0] - qx*X[2]);
  uv[2] = 2.0 * (qx*X[1] - qy*X[0]);

  X[0] += qw*uv[0] + (qy*uv[2] - qz*uv[1]) + tx;
  X[1] += qw*uv[1] + (qz*uv[0] - qx*uv[2]) + ty;
  X[2] += qw*uv[2] + (qx*uv[1] - qy*uv[0]) + tz;
}

__device__ __forceinline__ void
pinhole_jacobians(const float p[3], const float fx, const float fy, float Ju[6], float Jv[6], float Jz[6]) {
  const float X1=p[0], Y1=p[1], Z1=p[2];
  const float d = 1.0 / Z1; 
  const float d2 = d * d;

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
}


__global__ void dense_se3_forward_kernel(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> transforms,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> embeddings,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
  torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Hx,
  torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> bx,
  int radius)
{

  int batch_id = blockIdx.x; // batch_index
  int tx = threadIdx.x;
  int ix = blockIdx.y * NUM_THREADS + tx; // image_index

  const int ht = transforms.size(2);
  const int wd = transforms.size(3);
  const int ae_dim = embeddings.size(1);

  const int dim = ht * wd;
  const int h1 = ix / wd;
  const int w1 = ix % wd;

  const float* Xdata = points[batch_id].data();
  const float* rdata = targets[batch_id].data();
  const float* wdata = weights[batch_id].data();
  const float* ae_data = embeddings[batch_id].data();

  __shared__ float fx, fy, cx, cy;
  if (tx == 0) {
    fx = intrinsics[batch_id][0];
    fy = intrinsics[batch_id][1];
    cx = intrinsics[batch_id][2];
    cy = intrinsics[batch_id][3];
  }

  // transformation
  float G[7];
  float ae1[AE_DIM];

  // linear system
  float H[6][6], b[6];

  if (ix < dim) {
    G[0] = transforms[batch_id][0][h1][w1]; // tx
    G[1] = transforms[batch_id][1][h1][w1]; // ty
    G[2] = transforms[batch_id][2][h1][w1]; // tz
    G[3] = transforms[batch_id][3][h1][w1]; // qx
    G[4] = transforms[batch_id][4][h1][w1]; // qy
    G[5] = transforms[batch_id][5][h1][w1]; // qz
    G[6] = transforms[batch_id][6][h1][w1]; // qw

    for (int ii=0; ii<ae_dim; ii++) {
      ae1[ii] = embeddings[batch_id][ii][h1][w1];
    }
  }

  for (int ii=0; ii<6; ii++) {
    b[ii] = 0;
  }

  for (int ii=0; ii<6; ii++) {
    for (int jj=0; jj<6; jj++) {
      H[ii][jj] = 0;
    }
  }

  // jacobians
  float Ju[6], Jv[6], Jz[6];

  __shared__ float X0[3][NUM_THREADS];
  __shared__ float ae2[AE_DIM][NUM_THREADS];
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

      for (int k=0; k<ae_dim; k++)
        ae2[k][tx] = ae_data[jx + k*dim];
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

        float p[3] = { X0[0][j], X0[1][j], X0[2][j] };        
        se3_transform_point_inplace(G, p);
        
        // residual vectors
        const float X1=p[0], Y1=p[1], Z1=p[2];


        const float u = fx * (X1 / Z1) + cx;
        const float v = fy * (Y1 / Z1) + cy;
        const float ru = rvec[0][j] - u;
        const float rv = rvec[1][j] - v;
        const float rz = rvec[2][j] - 1.0 / Z1;

        // exclude pixels too close or errors too big
        if (Z1 < 0.1 || abs(ru) > 128 || abs(rv) > 128) 
          continue;
      
        float s=0.0;
        for (int k=0; k<ae_dim; k++) {
          s += (ae1[k] - ae2[k][j]) * (ae1[k] - ae2[k][j]);
        }
        
        const float w = sigmoid(-s);
        const float wu = w * wvec[0][j];
        const float wv = w * wvec[1][j];
        const float wz = w * wvec[2][j];

        pinhole_jacobians(p, fx, fy, Ju, Jv, Jz);

        for (int ii=0; ii<6; ii++) {
          b[ii] += wu*ru*Ju[ii] + wv*rv*Jv[ii] + wz*rz*Jz[ii];
        }

        for (int ii=0; ii<6; ii++) {
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
    }

    for (int ii=0; ii<6; ii++) {
      for (int jj=0; jj<6; jj++) {
        Hx[batch_id][ii][jj][h1][w1] = H[ii][jj];
      }
    }

  }
}


__global__ void dense_se3_backward_kernel1(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> transforms,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> embeddings,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Hx_grad,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> bx_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> embedding_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights_grad,
  int radius)
{

  int batch_id = blockIdx.x; // batch_index
  int tx = threadIdx.x;
  int ix = blockIdx.y * NUM_THREADS + tx; // image_index

  const int ht = transforms.size(2);
  const int wd = transforms.size(3);
  const int dim = ht * wd;
  const int ae_dim = embeddings.size(1);

  int h2 = ix / wd;
  int w2 = ix % wd;

  const float* transform_data = transforms[batch_id].data();
  const float* ae_data = embeddings[batch_id].data();
  const float* diffH_data = Hx_grad[batch_id].data();
  const float* diffb_data = bx_grad[batch_id].data();

  __shared__ float fx, fy, cx, cy;
  if (tx == 0) {
    fx = intrinsics[batch_id][0];
    fy = intrinsics[batch_id][1];
    cx = intrinsics[batch_id][2];
    cy = intrinsics[batch_id][3];
  }

  float X0[3];
  float target_u, target_v, target_z;
  float wu, wv, wz;

  float ae2[AE_DIM];
  float diff_ae2[AE_DIM];

  if (ix < dim) {
    X0[0] = points[batch_id][0][h2][w2];
    X0[1] = points[batch_id][1][h2][w2];
    X0[2] = points[batch_id][2][h2][w2];

    target_u = targets[batch_id][0][h2][w2];
    target_v = targets[batch_id][1][h2][w2];
    target_z = targets[batch_id][2][h2][w2];

    wu = weights[batch_id][0][h2][w2];
    wv = weights[batch_id][1][h2][w2];
    wz = weights[batch_id][2][h2][w2];
    
    for (int ii=0; ii<ae_dim; ii++) {
      ae2[ii] = ae_data[ix + ii*dim];
      diff_ae2[ii] = 0;
    }
  }

  // jacobians
  float Ju[6], Jv[6], Jz[6];
  float diff_ru = 0;
  float diff_rv = 0;
  float diff_rz = 0;
  float diff_wu = 0;
  float diff_wv = 0;
  float diff_wz = 0;

  __shared__ float Gs[NUM_THREADS][7];
  __shared__ float dH[6][6][NUM_THREADS];
  __shared__ float db[6][NUM_THREADS];
  __shared__ float ae1[AE_DIM][NUM_THREADS];
  
  __syncthreads();

  for (int i=0; i<dim; i+=NUM_THREADS) {
    int jx = i + tx;

    // read from global
    if (jx < dim) {
      Gs[tx][0] = transform_data[jx + 0*dim];
      Gs[tx][1] = transform_data[jx + 1*dim];
      Gs[tx][2] = transform_data[jx + 2*dim];
      Gs[tx][3] = transform_data[jx + 3*dim];
      Gs[tx][4] = transform_data[jx + 4*dim];
      Gs[tx][5] = transform_data[jx + 5*dim];
      Gs[tx][6] = transform_data[jx + 6*dim];

      for (int ii=0; ii<ae_dim; ii++) {
        ae1[ii][tx] = ae_data[jx + ii*dim];
      }

      for (int ii=0; ii<6; ii++) {
        for (int jj=0; jj<6; jj++) {
          dH[ii][jj][tx] = diffH_data[jx + (ii*6+jj)*dim];
        }
      }

      for (int ii=0; ii<6; ii++) {
        db[ii][tx] = diffb_data[jx + ii*dim];
      }
    }

    __syncthreads();

    for (int j=0; j<NUM_THREADS; j++) {
      jx = i + j;
      if (ix<dim && jx<dim) {
        int h1 = jx / wd;
        int w1 = jx % wd;

        int r = max(abs(h1-h2), abs(w1-w2));
        if (r > radius) continue;
  
        float p[3] = { X0[0], X0[1], X0[2] };
        se3_transform_point_inplace(&Gs[j][0], p);
        
        // residual vectors
        const float X1=p[0], Y1=p[1], Z1=p[2];
        const float u = fx * (X1 / Z1) + cx;
        const float v = fy * (Y1 / Z1) + cy;
        const float ru = target_u - u;
        const float rv = target_v - v;
        const float rz = target_z - 1.0 / Z1;

        float s=0.0;
        for (int k=0; k<ae_dim; k++) {
          s += (ae1[k][j] - ae2[k]) * (ae1[k][j] - ae2[k]);
        }
        
        float diff_w = 0.0f;
        const float w = sigmoid(-s);

        // exclude pixels too close or errors too big
        if (Z1 < 0.1 || abs(ru) > 128 || abs(rv) > 128) 
          continue;

        pinhole_jacobians(p, fx, fy, Ju, Jv, Jz);

        for (int ii=0; ii<6; ii++) {

          const float db_i = db[ii][j];
          // residual gradients
          diff_ru += w*wu*Ju[ii] * db_i;
          diff_rv += w*wv*Jv[ii] * db_i;
          diff_rz += w*wz*Jz[ii] * db_i;

          // weights gradients
          diff_wu += w*ru*Ju[ii] * db_i;
          diff_wv += w*rv*Jv[ii] * db_i;
          diff_wz += w*rz*Jz[ii] * db_i;

          // embedding weight
          diff_w += (wu*ru*Ju[ii] + wv*rv*Jv[ii] + wz*rz*Jz[ii]) * db_i;

          for (int jj=0; jj<6; jj++) {
            const float dH_ij = dH[ii][jj][j];
            diff_wu += w*Ju[ii]*Ju[jj] * dH_ij;
            diff_wv += w*Jv[ii]*Jv[jj] * dH_ij;
            diff_wz += w*Jz[ii]*Jz[jj] * dH_ij;
            diff_w += (wu*Ju[ii]*Ju[jj] + wv*Jv[ii]*Jv[jj] + wz*Jz[ii]*Jz[jj]) * dH_ij;
          }
        }

        float diff_s = -diff_w * sigmoid(-s) * (1.0f - sigmoid(-s));
        for (int k=0; k<ae_dim; k++) {
          diff_ae2[k] += -2 * diff_s * (ae1[k][j] - ae2[k]);
        }
      }
    }
    __syncthreads();
  }

  if (ix < dim) {
    targets_grad[batch_id][0][h2][w2] = diff_ru;
    targets_grad[batch_id][1][h2][w2] = diff_rv;
    targets_grad[batch_id][2][h2][w2] = diff_rz;

    weights_grad[batch_id][0][h2][w2] = diff_wu;
    weights_grad[batch_id][1][h2][w2] = diff_wv;
    weights_grad[batch_id][2][h2][w2] = diff_wz;

    for (int k=0; k<ae_dim; k++)
      embedding_grad[batch_id][k][h2][w2] += diff_ae2[k];
  }
}


__global__ void dense_se3_backward_kernel2(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> transforms,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> embeddings,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Hx_grad,
  const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> bx_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> embedding_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> targets_grad,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weights_grad,
  int radius) {

  int batch_id = blockIdx.x; // batch_index
  int tx = threadIdx.x;
  int ix = blockIdx.y * NUM_THREADS + tx; // image_index

  const int ht = transforms.size(2);
  const int wd = transforms.size(3);
  const int ae_dim = embeddings.size(1);

  const int dim = ht * wd;
  const int h1 = ix / wd;
  const int w1 = ix % wd;

  const float* transform_data = transforms[batch_id].data();
  const float* Xdata = points[batch_id].data();
  const float* rdata = targets[batch_id].data();
  const float* wdata = weights[batch_id].data();
  const float* ae_data = embeddings[batch_id].data();

  __shared__ float fx, fy, cx, cy;
  if (tx == 0) {
    fx = intrinsics[batch_id][0];
    fy = intrinsics[batch_id][1];
    cx = intrinsics[batch_id][2];
    cy = intrinsics[batch_id][3];
  }

  // transformation
  float G[7];
  float ae1[AE_DIM];
  float diff_ae1[AE_DIM];

  float db[6], dH[6][6];
  if (ix < dim) {
    G[0] = transform_data[ix + 0*dim]; // tx
    G[1] = transform_data[ix + 1*dim]; // ty
    G[2] = transform_data[ix + 2*dim]; // tz
    G[3] = transform_data[ix + 3*dim]; // qx
    G[4] = transform_data[ix + 4*dim]; // qy
    G[5] = transform_data[ix + 5*dim]; // qz
    G[6] = transform_data[ix + 6*dim]; // qw

    for (int ii=0; ii<ae_dim; ii++) {
      ae1[ii] = embeddings[batch_id][ii][h1][w1];
      diff_ae1[ii] = 0;
    }
    
    for (int ii=0; ii<6; ii++) {
      db[ii] = bx_grad[batch_id][ii][0][h1][w1];
    }

    for (int ii=0; ii<6; ii++) {
      for (int jj=0; jj<6; jj++) {
        dH[ii][jj] = Hx_grad[batch_id][ii][jj][h1][w1];
      }
    }
  }


  // jacobians
  float Ju[6], Jv[6], Jz[6];

  __shared__ float X0[3][NUM_THREADS];
  __shared__ float ae2[AE_DIM][NUM_THREADS];
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

      for (int k=0; k<ae_dim; k++)
        ae2[k][tx] = ae_data[jx + k*dim];
    }

    __syncthreads();

    for (int j=0; j<NUM_THREADS; j++) {
      jx = i + j;
      if (ix<dim && jx<dim) {
        int h2 = jx / wd;
        int w2 = jx % wd;

        int r = max(abs(h1-h2), abs(w1-w2));        
        if (r > radius) continue;

        float p[3] = { X0[0][j], X0[1][j], X0[2][j] };        
        se3_transform_point_inplace(G, p);
        
        // residual vectors
        const float X1=p[0], Y1=p[1], Z1=p[2];
        const float u = fx * (X1 / Z1) + cx;
        const float v = fy * (Y1 / Z1) + cy;
        
        const float ru = rvec[0][j] - u;
        const float rv = rvec[1][j] - v;
        const float rz = rvec[2][j] - 1.0 / Z1;

        float s=0.0;
        for (int k=0; k<ae_dim; k++) {
          s += (ae1[k] - ae2[k][j]) * (ae1[k] - ae2[k][j]);
        }
        
        const float w = sigmoid(-s);
        float diff_w = 0;

        const float wu = wvec[0][j];
        const float wv = wvec[1][j];
        const float wz = wvec[2][j];

        // exclude pixels too close or errors too big
        if (Z1 < 0.1 || abs(ru) > 128 || abs(rv) > 128) 
          continue;

        pinhole_jacobians(p, fx, fy, Ju, Jv, Jz);

        for (int ii=0; ii<6; ii++) {
          diff_w += (wu*ru*Ju[ii] + wv*rv*Jv[ii] + wz*rz*Jz[ii]) * db[ii];
          for (int jj=0; jj<6; jj++) {
            diff_w += (wu*Ju[ii]*Ju[jj] + wv*Jv[ii]*Jv[jj] + wz*Jz[ii]*Jz[jj]) * dH[ii][jj];
          }
        }


        float diff_s = -diff_w * sigmoid(-s) * (1.0f - sigmoid(-s));

        for (int k=0; k<ae_dim; k++) {
          diff_ae1[k] += 2 * diff_s * (ae1[k] - ae2[k][j]);
        }
      }
    }
    __syncthreads();
  }

  if (ix < dim) {
    for (int k=0; k<ae_dim; k++)
      embedding_grad[batch_id][k][h1][w1] += diff_ae1[k];
  }
}



std::vector<torch::Tensor> dense_se3_forward_cuda(
  torch::Tensor transforms,
  torch::Tensor embeddings,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  int radius)
{

  int batch_size = transforms.size(0);
  int ht = transforms.size(2);
  int wd = transforms.size(3);

  dim3 grid = dim3(batch_size, (ht*wd + NUM_THREADS-1) / NUM_THREADS);

  auto opts = targets.options();
  torch::Tensor H = torch::zeros({batch_size, 6, 6, ht, wd}, opts);
  torch::Tensor b = torch::zeros({batch_size, 6, 1, ht, wd}, opts);

  dense_se3_forward_kernel<<<grid, NUM_THREADS>>>(
    transforms.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    embeddings.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    H.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

  return {H, b};
}


std::vector<torch::Tensor> dense_se3_backward_cuda(
  torch::Tensor transforms,
  torch::Tensor embeddings,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  torch::Tensor H_grad,
  torch::Tensor b_grad,
  int radius)
{

  int batch_size = transforms.size(0);
  int ht = transforms.size(2);
  int wd = transforms.size(3);

  dim3 grid = dim3(batch_size, (ht*wd + NUM_THREADS-1) / NUM_THREADS);
  torch::Tensor embedding_grad = torch::zeros_like(embeddings);
  torch::Tensor targets_grad = torch::zeros_like(targets);
  torch::Tensor weights_grad = torch::zeros_like(weights);

  // backward pass split into two kernels to avoid atomics

  dense_se3_backward_kernel1<<<grid, NUM_THREADS>>>(
    transforms.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    embeddings.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    H_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    embedding_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    radius);

  dense_se3_backward_kernel2<<<grid, NUM_THREADS>>>(
    transforms.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    embeddings.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    H_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    b_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    embedding_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    targets_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    radius);

  return {embedding_grad, targets_grad, weights_grad};
}