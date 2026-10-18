#include "owned_g2o_optimizer.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <iostream>
#include <stdexcept>

namespace {
int vertices = 0, edges = 0, kernels = 0, algorithms = 0;
struct Vertex : g2o::VertexSE3Expmap { ~Vertex() override { ++vertices; } };
struct Edge : g2o::EdgeSE3ProjectXYZOnlyPose { ~Edge() override { ++edges; } };
struct Kernel : g2o::RobustKernelHuber { ~Kernel() override { ++kernels; } };
struct Algorithm : g2o::OptimizationAlgorithmLevenberg {
    Algorithm() : g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<g2o::BlockSolver_6_3>(
            std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>())) {}
    ~Algorithm() override { ++algorithms; }
};
void build(bool fail) {
    pyslam::OwnedG2oOptimizer optimizer;
    optimizer.setAlgorithm(optimizer.make<Algorithm>());
    auto *vertex = optimizer.make<Vertex>();
    vertex->setId(0);
    optimizer.addVertex(vertex);
    auto *removed = optimizer.make<Edge>();
    removed->setVertex(0, vertex);
    removed->setRobustKernel(optimizer.make<Kernel>());
    optimizer.addEdge(removed);
    removed->setRobustKernel(nullptr);
    optimizer.removeEdge(removed);
    auto *retained = optimizer.make<Edge>();
    retained->setVertex(0, vertex);
    retained->setRobustKernel(optimizer.make<Kernel>());
    optimizer.addEdge(retained);
    if (fail)
        throw std::runtime_error("Simulated optimization failure");
}
}

int main() {
    for (int i = 0; i < 100; ++i) {
        try { build(i % 2); } catch (const std::runtime_error &) {}
        const int count = i + 1;
        if (vertices != count || algorithms != count || edges != 2 * count || kernels != 2 * count) {
            std::cerr << "Native graph allocations were not destroyed exactly once: "
                      << vertices << ' ' << edges << ' ' << kernels << ' ' << algorithms << '\n';
            return 1;
        }
    }
    std::cout << "Native g2o ownership: repeated graphs, detached kernels, removed edges, exceptions OK\n";
}
