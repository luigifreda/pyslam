#pragma once

#include <g2o/config.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyslam {

// g2opy disables g2o's implicit ownership so Python can own graph objects.
// Native graphs have no Python owners: retain every allocation, including
// removed edges and kernels detached before the final optimization rounds.
class OwnedG2oOptimizer : public g2o::SparseOptimizer {
  public:
    template <class T, class... Args> T *make(Args &&...args) {
        auto object = std::make_unique<T>(std::forward<Args>(args)...);
        auto *result = object.get();
#if G2O_DELETE_IMPLICITLY_OWNED_OBJECTS
        object.release(); // Standard g2o owns the graph and attached kernels.
#else
        if constexpr (std::is_base_of_v<g2o::OptimizationAlgorithm, T>)
            algorithms_.push_back(std::move(object));
        else if constexpr (std::is_base_of_v<g2o::RobustKernel, T>)
            kernels_.push_back(std::move(object));
        else if constexpr (std::is_base_of_v<g2o::HyperGraph::Edge, T>)
            edges_.push_back(std::move(object));
        else {
            static_assert(std::is_base_of_v<g2o::HyperGraph::Vertex, T>);
            vertices_.push_back(std::move(object));
        }
#endif
        return result;
    }

    ~OwnedG2oOptimizer() override {
#if !G2O_DELETE_IMPLICITLY_OWNED_OBJECTS
        setAlgorithm(nullptr);
        clear();
        algorithms_.clear();
        edges_.clear();
        vertices_.clear();
        kernels_.clear();
#endif
    }

  private:
#if !G2O_DELETE_IMPLICITLY_OWNED_OBJECTS
    std::vector<std::unique_ptr<g2o::OptimizationAlgorithm>> algorithms_;
    std::vector<std::unique_ptr<g2o::RobustKernel>> kernels_;
    std::vector<std::unique_ptr<g2o::HyperGraph::Edge>> edges_;
    std::vector<std::unique_ptr<g2o::HyperGraph::Vertex>> vertices_;
#endif
};

} // namespace pyslam
