/**
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/internal/NonlinearOptimizerState.h>

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/base/timing.h>

#include <boost/format.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using namespace gtsam;
using boost::adaptors::map_values;

class LevenbergMarquardtOptimizerG2o : public NonlinearOptimizer
{
protected:
    const LevenbergMarquardtParams params_;
    boost::posix_time::ptime startTime_;
    double _tau = 1e-5;

    // Internal LM state for more freely mimicking g2o algorithm
    struct G2oLmState
    {
        double ni = 2.0;
        double currentLambda;
        int maxTrialsAfterFailure;        
    } _g2oLmState;

    struct State : public internal::NonlinearOptimizerState
    {
        double lambda;
        double currentFactor;
        int totalNumberInnerIterations;

        State(const Values &initialValues, double error, double lambda, double factor,
              unsigned int iterations = 0, unsigned int totalInner = 0)
            : internal::NonlinearOptimizerState(initialValues, error, iterations),
              lambda(lambda), currentFactor(factor), totalNumberInnerIterations(totalInner) {}

        void increaseLambda(const LevenbergMarquardtParams &params)
        {
            lambda *= currentFactor;
            totalNumberInnerIterations++;
            if (!params.useFixedLambdaFactor)
                currentFactor *= 2.0;
        }

        std::unique_ptr<State> decreaseLambda(const LevenbergMarquardtParams &params,
                                              double rho, Values &&newValues, double newError) const
        {
            double newLambda = lambda;
            double newFactor = currentFactor;
            if (params.useFixedLambdaFactor)
            {
                newLambda /= currentFactor;
            }
            else
            {
                newLambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
                newFactor *= 2.0;
            }
            newLambda = std::max(params.lambdaLowerBound, newLambda);
            return std::make_unique<State>(std::move(newValues), newError, newLambda, newFactor,
                                           iterations + 1, totalNumberInnerIterations + 1);
        }
    };

public:
    LevenbergMarquardtOptimizerG2o(const NonlinearFactorGraph &graph,
                                   const Values &initialValues,
                                   const LevenbergMarquardtParams &params = LevenbergMarquardtParams())
        : NonlinearOptimizer(graph, std::make_unique<State>(initialValues, graph.error(initialValues),
                                                            params.lambdaInitial, params.lambdaFactor)),
          params_(LevenbergMarquardtParams::EnsureHasOrdering(params, graph))
    {
        _g2oLmState.currentLambda = computeLambdaInit();
        _g2oLmState.maxTrialsAfterFailure = params.maxIterations;
    }

    LevenbergMarquardtOptimizerG2o(const NonlinearFactorGraph &graph,
                                   const Values &initialValues,
                                   const Ordering &ordering,
                                   const LevenbergMarquardtParams &params = LevenbergMarquardtParams())
        : NonlinearOptimizer(graph, std::make_unique<State>(initialValues, graph.error(initialValues),
                                                            params.lambdaInitial, params.lambdaFactor)),
          params_(LevenbergMarquardtParams::ReplaceOrdering(params, ordering))
    {
        _g2oLmState.currentLambda = computeLambdaInit();
        _g2oLmState.maxTrialsAfterFailure = params.maxIterations;
    }

    double computeLambdaInit() const
    {
        auto linear = graph_.linearize(state_->values);
        VectorValues diag = linear->hessianDiagonal();
        double maxDiag = 0.0;
        for (const auto &[_, vec] : diag)
            for (size_t j = 0; j < vec.size(); ++j)
                maxDiag = std::max(maxDiag, std::abs(vec[j]));
        return _tau * maxDiag;
    }

    double computeScale(const VectorValues &delta, const VectorValues &b) const
    {
        double scale = 0.0;
        for (const auto &[key, dx] : delta)
        {
            const auto &bval = b.at(key);
            scale += dx.dot(_g2oLmState.currentLambda * dx + bval);
        }
        return scale + 1e-3;
    }

    GaussianFactorGraph buildDampedSystem(const GaussianFactorGraph &linear, const VectorValues &sqrtHessianDiagonal) const
    {
        auto currentState = static_cast<const State *>(state_.get());
        GaussianFactorGraph damped = linear;
        for (const auto &[key, diagVec] : sqrtHessianDiagonal)
        {
            Vector diag = diagVec.cwiseSqrt();
            Matrix A = diag.asDiagonal();
            Vector b = Vector::Zero(diag.size());
            SharedDiagonal model = noiseModel::Isotropic::Sigma(diag.size(), 1.0 / std::sqrt(currentState->lambda));
            damped.push_back(boost::make_shared<JacobianFactor>(key, A, b, model));
        }
        return damped;
    }

    GaussianFactorGraph::shared_ptr iterate() override
    {
        auto currentState = static_cast<State *>(state_.get());
        gttic(LM_iterate);

        auto linear = graph_.linearize(currentState->values);
        VectorValues sqrtHessianDiagonal = linear->hessianDiagonal();
        VectorValues b = linear->gradientAtZero();

        if (currentState->totalNumberInnerIterations == 0)
        { 
            _g2oLmState.currentLambda = computeLambdaInit();

            // write initial error
            writeLogFile(currentState->error);
    
            if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
            {
                std::cout << "Initial error: " << currentState->error
                          << ", values: " << currentState->values.size() << std::endl;
            }
        }

        double currentChi = currentState->error;
        double tempChi = currentChi;
        double rho = 0.0;
        int qmax = 0;

        do
        {

        #ifdef GTSAM_USING_NEW_BOOST_TIMERS
            boost::timer::cpu_timer lamda_iteration_timer;
            lamda_iteration_timer.start();
        #else
            boost::timer lamda_iteration_timer;
            lamda_iteration_timer.restart();
        #endif

        
            auto damped = buildDampedSystem(*linear, sqrtHessianDiagonal);
            VectorValues delta;
            bool success = true;
            try
            {
                delta = solve(damped, params_);
            }
            catch (...)
            {
                success = false;
                tempChi = std::numeric_limits<double>::max();
            }

            if (success)
            {
                Values newValues = currentState->values.retract(delta);
                tempChi = graph_.error(newValues);
                rho = (currentChi - tempChi) / computeScale(delta, b);

                if (rho > 0 && std::isfinite(tempChi))
                {
                    _g2oLmState.currentLambda *= std::max(1.0 / 3.0, std::min(2.0 / 3.0, 1.0 - std::pow((2.0 * rho - 1.0), 3)));
                    _g2oLmState.ni = 2.0;
                    currentChi = tempChi;
                    state_ = currentState->decreaseLambda(params_, rho, std::move(newValues), tempChi);
                }
                else
                {
                    _g2oLmState.currentLambda *= _g2oLmState.ni;
                    _g2oLmState.ni *= 2.0;
                }
            }
            else
            {
                _g2oLmState.currentLambda *= _g2oLmState.ni;
                _g2oLmState.ni *= 2.0;
            }

            if (!std::isfinite(_g2oLmState.currentLambda))
                break;
            qmax++;

            if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
            {
        // do timing
        #ifdef GTSAM_USING_NEW_BOOST_TIMERS
                double iterationTime = 1e-9 * lamda_iteration_timer.elapsed().wall;
        #else
                double iterationTime = lamda_iteration_timer.elapsed();
        #endif
                if (currentState->iterations == 0)
                    std::cout << "iter      cost      cost_change    lambda  success iter_time" << std::endl;
                    std::cout << boost::format("% 4d % 8e   % 3.2e   % 3.2e  % 4d   % 3.2e") % currentState->iterations %
                            tempChi % rho % currentState->lambda % success %
                            iterationTime
                     << std::endl;
            }                    

        } while (rho < 0 && qmax < _g2oLmState.maxTrialsAfterFailure);

        return linear;
    }

    const LevenbergMarquardtParams &params() const { return params_; }

    double lambda() const
    {
        auto currentState = static_cast<const State *>(state_.get());
        return currentState->lambda;
    }

    int getInnerIterations() const
    {
        auto currentState = static_cast<const State *>(state_.get());
        return currentState->totalNumberInnerIterations;
    }

    void writeLogFile(double currentError)
    {
        auto currentState = static_cast<const State *>(state_.get());
        if (!params_.logFile.empty())
        {
            std::ofstream os(params_.logFile, std::ios::app);
            boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
            os << currentState->totalNumberInnerIterations << ","
               << 1e-6 * (now - startTime_).total_microseconds() << ","
               << currentError << "," << currentState->lambda << ","
               << currentState->iterations << std::endl;
        }
    }

    GaussianFactorGraph::shared_ptr linearize() const
    {
        return graph_.linearize(state_->values);
    }

protected:
    /** Access the parameters (base class version) */
    const NonlinearOptimizerParams &_params() const override
    {
        return params_;
    }    
};
