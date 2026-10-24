#pragma once

#include <g2o/config.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/hyper_graph_action.h>
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyslam {

// g2o accepts bool*, which must only be written by the solver thread. Native mapping
// submits an atomic request; an iteration action transfers it onto that local bool.
// A scoped thread-local context preserves the existing public bool* optimizer API.
class ScopedAtomicOptimizationAbort {
  public:
    inline static thread_local const std::atomic<bool> *current = nullptr;
    explicit ScopedAtomicOptimizationAbort(const std::atomic<bool> &flag)
        : previous_(current) { current = &flag; }
    ~ScopedAtomicOptimizationAbort() { current = previous_; }
  private:
    const std::atomic<bool> *previous_;
};

// g2opy disables g2o's implicit ownership so Python can own graph objects.
// Native graphs have no Python owners: retain every allocation, including
// removed edges and kernels detached before the final optimization rounds.
class OwnedG2oOptimizer : public g2o::SparseOptimizer {
  public:
    void setForceStopFlag(bool *flag) {
        g2o::SparseOptimizer::setForceStopFlag(flag);
        if (flag && ScopedAtomicOptimizationAbort::current) {
            atomic_abort_.source = ScopedAtomicOptimizationAbort::current;
            atomic_abort_.destination = flag;
            *flag = atomic_abort_.source->load();
            addPreIterationAction(&atomic_abort_);
        }
    }

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
        removePreIterationAction(&atomic_abort_);
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
    struct AtomicAbortAction : g2o::HyperGraphAction {
        const std::atomic<bool> *source = nullptr;
        bool *destination = nullptr;
        g2o::HyperGraphAction *operator()(const g2o::HyperGraph *,
                                         g2o::HyperGraphAction::Parameters *) override {
            *destination = *destination || source->load();
            return this;
        }
    } atomic_abort_;
#if !G2O_DELETE_IMPLICITLY_OWNED_OBJECTS
    std::vector<std::unique_ptr<g2o::OptimizationAlgorithm>> algorithms_;
    std::vector<std::unique_ptr<g2o::RobustKernel>> kernels_;
    std::vector<std::unique_ptr<g2o::HyperGraph::Edge>> edges_;
    std::vector<std::unique_ptr<g2o::HyperGraph::Vertex>> vertices_;
#endif
};

} // namespace pyslam
