#include <torch/extension.h>
#include <vector>


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// CUDA forward declarations
std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> dense_se3_forward_cuda(
  torch::Tensor transforms,
  torch::Tensor embeddings,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  int radius);

std::vector<torch::Tensor> dense_se3_backward_cuda(
  torch::Tensor transforms,
  torch::Tensor embeddings,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  torch::Tensor H_grad,
  torch::Tensor b_grad,
  int radius);


std::vector<torch::Tensor> se3_build_cuda(
  torch::Tensor attention,
  torch::Tensor transforms,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  int radius);


std::vector<torch::Tensor> se3_build_backward_cuda(
  torch::Tensor attention,
  torch::Tensor transforms,
  torch::Tensor points,
  torch::Tensor targets,
  torch::Tensor weights,
  torch::Tensor intrinsics,
  torch::Tensor H_grad,
  torch::Tensor b_grad,
  int radius);


std::vector<torch::Tensor> cholesky_solve6x6_forward_cuda(
  torch::Tensor H, torch::Tensor b);

std::vector<torch::Tensor> cholesky_solve6x6_backward_cuda(
  torch::Tensor H, torch::Tensor b, torch::Tensor dx);

// c++ python binding
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}


std::vector<torch::Tensor> se3_build(
    torch::Tensor attention,
    torch::Tensor transforms,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    int radius) {

  CHECK_INPUT(transforms);
  CHECK_INPUT(attention);
  CHECK_INPUT(points);
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(intrinsics);

  return se3_build_cuda(attention, transforms, 
    points, targets, weights, intrinsics, radius);
}

std::vector<torch::Tensor> se3_build_backward(
    torch::Tensor attention,
    torch::Tensor transforms,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    torch::Tensor H_grad,
    torch::Tensor b_grad,
    int radius) {

  CHECK_INPUT(transforms);
  CHECK_INPUT(attention);
  CHECK_INPUT(points);
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(intrinsics);

  CHECK_INPUT(H_grad);
  CHECK_INPUT(b_grad);

  return se3_build_backward_cuda(attention, transforms, points, 
    targets, weights, intrinsics, H_grad, b_grad, radius);
}

std::vector<torch::Tensor> se3_build_inplace(
    torch::Tensor transforms,
    torch::Tensor embeddings,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    int radius) {

  CHECK_INPUT(transforms);
  CHECK_INPUT(embeddings);
  CHECK_INPUT(points);
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(intrinsics);

  return dense_se3_forward_cuda(transforms, embeddings, 
    points, targets, weights, intrinsics, radius);
}

std::vector<torch::Tensor> se3_build_inplace_backward(
    torch::Tensor transforms,
    torch::Tensor embeddings,
    torch::Tensor points,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor intrinsics,
    torch::Tensor H_grad,
    torch::Tensor b_grad,
    int radius) {

  CHECK_INPUT(transforms);
  CHECK_INPUT(embeddings);
  CHECK_INPUT(points);
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(intrinsics);

  CHECK_INPUT(H_grad);
  CHECK_INPUT(b_grad);

  return dense_se3_backward_cuda(transforms, embeddings, points, 
    targets, weights, intrinsics, H_grad, b_grad, radius);
}


std::vector<torch::Tensor> cholesky6x6_forward(
    torch::Tensor H,
    torch::Tensor b) {
  CHECK_INPUT(H);
  CHECK_INPUT(b);
  
  return cholesky_solve6x6_forward_cuda(H, b);
}

std::vector<torch::Tensor> cholesky6x6_backward(
    torch::Tensor H,
    torch::Tensor b,
    torch::Tensor dx) {

  CHECK_INPUT(H);
  CHECK_INPUT(b);
  CHECK_INPUT(dx);

  return cholesky_solve6x6_backward_cuda(H, b, dx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");

  // RAFT-3D functions
  m.def("se3_build", &se3_build, "build forward");
  m.def("se3_build_backward", &se3_build_backward, "build backward");
  
  m.def("se3_build_inplace", &se3_build_inplace, "build forward");
  m.def("se3_build_inplace_backward", &se3_build_inplace_backward, "build backward");

  m.def("cholesky6x6_forward", &cholesky6x6_forward, "solve forward");
  m.def("cholesky6x6_backward", &cholesky6x6_backward, "solve backward");
}

