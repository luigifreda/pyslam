
#include "lietorch_cpu.h"
#include <Eigen/Dense>

#include <iostream>
#include "common.h"
#include "dispatch.h"

#include "so3.h"
#include "rxso3.h"
#include "se3.h"
#include "sim3.h"


template <typename Group, typename scalar_t>
void exp_forward_kernel(const scalar_t* a_ptr, scalar_t* X_ptr, int batch_size) {
    // exponential map forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;
    
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Tangent a(a_ptr + i*Group::K);
            Eigen::Map<Data>(X_ptr + i*Group::N) = Group::Exp(a).data();
        }
    });
}

template <typename Group, typename scalar_t>
void exp_backward_kernel(const scalar_t* grad, const scalar_t* a_ptr, scalar_t* da, int batch_size) {
    // exponential map backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Tangent a(a_ptr + i*Group::K);
            Grad dX(grad + i*Group::N);
            Eigen::Map<Grad>(da + i*Group::K) = dX * Group::left_jacobian(a);
        }
    });
}

template <typename Group, typename scalar_t>
void log_forward_kernel(const scalar_t* X_ptr, scalar_t* a_ptr, int batch_size) {
    // logarithm map forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Tangent a = Group(X_ptr + i*Group::N).Log();
            Eigen::Map<Tangent>(a_ptr + i*Group::K) = a;
        }
    });
}

template <typename Group, typename scalar_t>
void log_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, scalar_t* dX, int batch_size) {
    // logarithm map backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Tangent a = Group(X_ptr + i*Group::N).Log();
            Grad da(grad + i*Group::K);
            Eigen::Map<Grad>(dX + i*Group::N) = da * Group::left_jacobian_inverse(a);
        }
    });
}

template <typename Group, typename scalar_t>
void inv_forward_kernel(const scalar_t* X_ptr, scalar_t* Y_ptr, int batch_size) {
    // group inverse forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Eigen::Map<Data>(Y_ptr + i*Group::N) = X.inv().data();
        }
    });
}

template <typename Group, typename scalar_t>
void inv_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, scalar_t *dX, int batch_size) {
    // group inverse backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group Y = Group(X_ptr + i*Group::N).inv();
            Grad dY(grad + i*Group::N);
            Eigen::Map<Grad>(dX + i*Group::N) = -dY * Y.Adj();
        }
    });
}

template <typename Group, typename scalar_t>
void mul_forward_kernel(const scalar_t* X_ptr, const scalar_t* Y_ptr, scalar_t* Z_ptr, int batch_size) {
    // group multiplication forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group Z = Group(X_ptr + i*Group::N) * Group(Y_ptr + i*Group::N);
            Eigen::Map<Data>(Z_ptr + i*Group::N) = Z.data();
        }
    });
}

template <class Group, typename scalar_t>
void mul_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, const scalar_t* Y_ptr, scalar_t* dX, scalar_t* dY, int batch_size) {
    // group multiplication backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Grad dZ(grad + i*Group::N);
            Group X(X_ptr + i*Group::N);        
            Eigen::Map<Grad>(dX + i*Group::N) = dZ;
            Eigen::Map<Grad>(dY + i*Group::N) = dZ * X.Adj();
        }
    });
}

template <typename Group, typename scalar_t>
void adj_forward_kernel(const scalar_t* X_ptr, const scalar_t* a_ptr, scalar_t* b_ptr, int batch_size) {
    // adjoint forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Tangent a(a_ptr + i*Group::K);
            Eigen::Map<Tangent>(b_ptr + i*Group::K) = X.Adj(a);
        }
    });
}

template <typename Group, typename scalar_t>
void adj_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, const scalar_t* a_ptr, scalar_t* dX, scalar_t* da, int batch_size) {
    // adjoint backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Grad db(grad + i*Group::K);

            Tangent a(a_ptr + i*Group::K);
            Tangent b = X.Adj() * a;

            Eigen::Map<Grad>(da + i*Group::K) = db * X.Adj();
            Eigen::Map<Grad>(dX + i*Group::N) = -db * Group::adj(b);
        }
    });
}

template <typename Group, typename scalar_t>
void adjT_forward_kernel(const scalar_t* X_ptr, const scalar_t* a_ptr, scalar_t* b_ptr, int batch_size) {
    // adjoint forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Tangent a(a_ptr + i*Group::K);
            Eigen::Map<Tangent>(b_ptr + i*Group::K) = X.AdjT(a);
        }
    });
}

template <typename Group, typename scalar_t>
void adjT_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, const scalar_t* a_ptr, scalar_t* dX, scalar_t* da, int batch_size) {
    // adjoint backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);        
            Tangent db(grad + i*Group::K);
            Grad a(a_ptr + i*Group::K);

            Eigen::Map<Tangent>(da + i*Group::K) = X.Adj(db);
            Eigen::Map<Grad>(dX + i*Group::N) = -a * Group::adj(X.Adj(db));
        }
    });
}


template <typename Group, typename scalar_t>
void act_forward_kernel(const scalar_t* X_ptr, const scalar_t* p_ptr, scalar_t* q_ptr, int batch_size) {
    // action on point forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;
    using Point = Eigen::Matrix<scalar_t,3,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Point p(p_ptr + i*3);
            Eigen::Map<Point>(q_ptr + i*3) = X * p;
        }
    });
}

template <typename Group, typename scalar_t>
void act_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, const scalar_t* p_ptr, scalar_t* dX, scalar_t* dp, int batch_size) {
    // adjoint backward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;
    using Point = Eigen::Matrix<scalar_t,3,1>;
    using PointGrad = Eigen::Matrix<scalar_t,1,3>;
    using Transformation = Eigen::Matrix<scalar_t,4,4>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Point p(p_ptr + i*3);
            PointGrad dq(grad + i*3);

            Eigen::Map<PointGrad>(dp + i*3) = dq * X.Matrix().template block<3,3>(0,0);
            Eigen::Map<Grad>(dX + i*Group::N) = dq * Group::act_jacobian(X*p);
        }
    });
}


// template <typename Group, typename scalar_t>
// void tovec_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, scalar_t* dX, int batch_size) {
//     // group inverse forward kernel
//     using Data = Eigen::Matrix<scalar_t,Group::N,1>;
//     using Grad = Eigen::Matrix<scalar_t,1,Group::N>;

//     at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
//         for (int64_t i=start; i<end; i++) {
//             Group X(X_ptr + i*Group::N);
//             Grad g(grad + i*Group::N);
//             Eigen::Map<Grad>(dX + i*Group::N) = g * X.vec_jacobian();
//         }
//     });
// }

// template <typename Group, typename scalar_t>
// void fromvec_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, scalar_t* dX, int batch_size) {
//     // group inverse forward kernel
//     using Data = Eigen::Matrix<scalar_t,Group::N,1>;
//     using Grad = Eigen::Matrix<scalar_t,1,Group::N>;

//     at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
//         for (int64_t i=start; i<end; i++) {
//             Group X(X_ptr + i*Group::N);
//             Grad g(grad + i*Group::N);
//             Eigen::Map<Grad>(dX + i*Group::N) = g * X.vec_jacobian();
//         }
//     });
// }


template <typename Group, typename scalar_t>
void act4_forward_kernel(const scalar_t* X_ptr, const scalar_t* p_ptr, scalar_t* q_ptr, int batch_size) {
    // action on homogeneous point forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;
    using Point = Eigen::Matrix<scalar_t,4,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Point p(p_ptr + i*4);
            Eigen::Map<Point>(q_ptr + i*4) = X.act4(p);
        }
    });
}

template <typename Group, typename scalar_t>
void act4_backward_kernel(const scalar_t* grad, const scalar_t* X_ptr, const scalar_t* p_ptr, scalar_t* dX, scalar_t* dp, int batch_size) {
    // action on homogeneous point backward kernel

    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Grad = Eigen::Matrix<scalar_t,1,Group::K>;
    using Point = Eigen::Matrix<scalar_t,4,1>;
    using PointGrad = Eigen::Matrix<scalar_t,1,4>;
    using Transformation = Eigen::Matrix<scalar_t,4,4>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Point p(p_ptr + i*4);
            PointGrad dq(grad + i*4);

            Eigen::Map<PointGrad>(dp + i*4) = dq * X.Matrix4x4();
            const Point q = X.act4(p);
            Eigen::Map<Grad>(dX + i*Group::N) = dq * Group::act4_jacobian(q);
        }
    });
}

template <typename Group, typename scalar_t>
void as_matrix_forward_kernel(const scalar_t* X_ptr, scalar_t* T_ptr, int batch_size) {
    // group inverse forward kernel
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;
    using Matrix4 = Eigen::Matrix<scalar_t,4,4,Eigen::RowMajor>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Eigen::Map<Matrix4>(T_ptr + i*16) = X.Matrix4x4();
        }
    });
}

template <typename Group, typename scalar_t>
void orthogonal_projector_kernel(const scalar_t* X_ptr, scalar_t* P_ptr, int batch_size) {
    // group inverse forward kernel
    using Proj = Eigen::Matrix<scalar_t,Group::N,Group::N,Eigen::RowMajor>;
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Eigen::Map<Proj>(P_ptr + i*Group::N*Group::N) = X.orthogonal_projector();
        }
    });
}

template <typename Group, typename scalar_t>
void jleft_forward_kernel(const scalar_t* X_ptr, const scalar_t* a_ptr, scalar_t* b_ptr, int batch_size) {
    // left-jacobian inverse action
    using Tangent = Eigen::Matrix<scalar_t,Group::K,1>;
    using Data = Eigen::Matrix<scalar_t,Group::N,1>;

    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
        for (int64_t i=start; i<end; i++) {
            Group X(X_ptr + i*Group::N);
            Tangent a(a_ptr + i*Group::K);
            Tangent b = Group::left_jacobian_inverse(X.Log()) * a;
            Eigen::Map<Tangent>(b_ptr + i*Group::K) = b;
        }
    });
}

// unary operations

torch::Tensor exp_forward_cpu(int group_id, torch::Tensor a) {
    int batch_size = a.size(0);
    torch::Tensor X;

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, a.type(), "exp_forward_kernel", ([&] {
        X = torch::zeros({batch_size, group_t::N}, a.options());
        exp_forward_kernel<group_t, scalar_t>(
            a.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return X;
}

std::vector<torch::Tensor> exp_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor a) {
    int batch_size = a.size(0);
    torch::Tensor da = torch::zeros(a.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, a.type(), "exp_backward_kernel", ([&] {
        exp_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            da.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {da};
}

torch::Tensor log_forward_cpu(int group_id, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor a;

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "log_forward_kernel", ([&] {
        a = torch::zeros({batch_size, group_t::K}, X.options());
        log_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return a;
}

std::vector<torch::Tensor> log_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "log_backward_kernel", ([&] {
        log_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX};
}

torch::Tensor inv_forward_cpu(int group_id, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor Y = torch::zeros_like(X);

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "inv_forward_kernel", ([&] {
        inv_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            Y.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return Y;
}

std::vector<torch::Tensor> inv_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "inv_backward_kernel", ([&] {
        inv_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX};
}

// binary operations
torch::Tensor mul_forward_cpu(int group_id, torch::Tensor X, torch::Tensor Y) {
    int batch_size = X.size(0);
    torch::Tensor Z = torch::zeros_like(X);

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "mul_forward_kernel", ([&] {
        mul_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            Y.data_ptr<scalar_t>(), 
            Z.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return Z;
}

std::vector<torch::Tensor> mul_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X, torch::Tensor Y) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());
    torch::Tensor dY = torch::zeros(Y.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "mul_backward_kernel", ([&] {
        mul_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            Y.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            dY.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX, dY};
}

torch::Tensor adj_forward_cpu(int group_id, torch::Tensor X, torch::Tensor a) {
    int batch_size = X.size(0);
    torch::Tensor b = torch::zeros(a.sizes(), a.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "adj_forward_kernel", ([&] {
        adj_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            b.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return b;
}

std::vector<torch::Tensor> adj_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X, torch::Tensor a) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());
    torch::Tensor da = torch::zeros(a.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "adj_backward_kernel", ([&] {
        adj_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            da.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX, da};
}


torch::Tensor adjT_forward_cpu(int group_id, torch::Tensor X, torch::Tensor a) {
    int batch_size = X.size(0);
    torch::Tensor b = torch::zeros(a.sizes(), a.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "adjT_forward_kernel", ([&] {
        adjT_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            b.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return b;
}

std::vector<torch::Tensor> adjT_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X, torch::Tensor a) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());
    torch::Tensor da = torch::zeros(a.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "adjT_backward_kernel", ([&] {
        adjT_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            da.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX, da};
}


torch::Tensor act_forward_cpu(int group_id, torch::Tensor X, torch::Tensor p) {
    int batch_size = X.size(0);
    torch::Tensor q = torch::zeros(p.sizes(), p.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "act_forward_kernel", ([&] {
        act_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            p.data_ptr<scalar_t>(), 
            q.data_ptr<scalar_t>(),
            batch_size);
    }));

    return q;
}

std::vector<torch::Tensor> act_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X, torch::Tensor p) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());
    torch::Tensor dp = torch::zeros(p.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "act_backward_kernel", ([&] {
        act_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            p.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            dp.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX, dp};
}


torch::Tensor act4_forward_cpu(int group_id, torch::Tensor X, torch::Tensor p) {
    int batch_size = X.size(0);
    torch::Tensor q = torch::zeros(p.sizes(), p.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "act4_forward_kernel", ([&] {
        act4_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            p.data_ptr<scalar_t>(), 
            q.data_ptr<scalar_t>(),
            batch_size);
    }));

    return q;
}

std::vector<torch::Tensor> act4_backward_cpu(int group_id, torch::Tensor grad, torch::Tensor X, torch::Tensor p) {
    int batch_size = X.size(0);
    torch::Tensor dX = torch::zeros(X.sizes(), grad.options());
    torch::Tensor dp = torch::zeros(p.sizes(), grad.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "act4_backward_kernel", ([&] {
        act4_backward_kernel<group_t, scalar_t>(
            grad.data_ptr<scalar_t>(), 
            X.data_ptr<scalar_t>(), 
            p.data_ptr<scalar_t>(), 
            dX.data_ptr<scalar_t>(), 
            dp.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return {dX, dp};
}


torch::Tensor as_matrix_forward_cpu(int group_id, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor T4x4 = torch::zeros({X.size(0), 4, 4}, X.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "as_matrix_forward_kernel", ([&] {
        as_matrix_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            T4x4.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return T4x4;
}


torch::Tensor orthogonal_projector_cpu(int group_id, torch::Tensor X) {
    int batch_size = X.size(0);
    torch::Tensor P;
    
    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "orthogonal_projector_kernel", ([&] {
        P = torch::zeros({X.size(0), group_t::N, group_t::N}, X.options());
        orthogonal_projector_kernel<group_t, scalar_t>(X.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(), batch_size);
    }));

    return P;
}



torch::Tensor jleft_forward_cpu(int group_id, torch::Tensor X, torch::Tensor a) {
    int batch_size = X.size(0);
    torch::Tensor b = torch::zeros(a.sizes(), a.options());

    DISPATCH_GROUP_AND_FLOATING_TYPES(group_id, X.type(), "jleft_forward_kernel", ([&] {
        jleft_forward_kernel<group_t, scalar_t>(
            X.data_ptr<scalar_t>(), 
            a.data_ptr<scalar_t>(), 
            b.data_ptr<scalar_t>(), 
            batch_size);
    }));

    return b;
}