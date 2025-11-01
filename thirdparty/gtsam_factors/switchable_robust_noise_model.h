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

 // #define GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR  // before including gtsam
 
 #include "numerical_derivative.h"
 
 #include <gtsam/inference/Symbol.h>
 
 #include <gtsam/slam/BetweenFactor.h>
 
 #include <gtsam/geometry/Cal3DS2.h>
 #include <gtsam/geometry/Cal3_S2.h>
 #include <gtsam/geometry/Cal3_S2Stereo.h>
 #include <gtsam/geometry/PinholeCamera.h>
 #include <gtsam/geometry/Pose3.h>
 #include <gtsam/geometry/Similarity3.h>
 #include <gtsam/geometry/StereoCamera.h>
 #include <gtsam/geometry/StereoPoint2.h>
 
 #include <gtsam/linear/NoiseModel.h>
 #include <gtsam/nonlinear/NonlinearFactor.h>
 
 #include <gtsam/base/Matrix.h>
 #include <gtsam/base/Vector.h>
 
 #include <boost/shared_ptr.hpp> // Include Boost
 
 #include <iostream>
 
 using namespace gtsam;
 using symbol_shorthand::X;


 template <typename T> boost::shared_ptr<T> create_shared_noise_model(const T &model) {
    return boost::make_shared<T>(model); // This makes a copy of model
}

template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const boost::shared_ptr<T> &model) {
    return model; // No copy, just return the same shared_ptr
}

namespace gtsam_factors {
 
/**
 * @brief A class that allows for a switchable robust noise model
 * 
 * This class allows for a switchable robust noise model, which can be used to
 * switch between a robust noise model and a diagonal noise model.
 * 
 * @tparam T The type of the noise model
 * @param dim The dimension of the noise model
 * @param sigma The sigma of the noise model
 * @param huberThreshold The huber threshold of the noise model
 */ 
 class SwitchableRobustNoiseModel : public noiseModel::Base {
   private:
     SharedNoiseModel robustNoiseModel_;
     SharedNoiseModel diagonalNoiseModel_;
     SharedNoiseModel activeNoiseModel_; // Currently active model
 
   public:
     /// Constructor initializes both diagonal/isotropic and robust models
     SwitchableRobustNoiseModel(int dim, double sigma, double huberThreshold) : Base(dim) {
 
         // Create isotropic noise model
         diagonalNoiseModel_ = noiseModel::Isotropic::Sigma(dim, sigma);
 
         // Create robust noise model (Huber loss with isotropic base)
         auto robustKernel = noiseModel::mEstimator::Huber::Create(huberThreshold);
         robustNoiseModel_ = noiseModel::Robust::Create(robustKernel, diagonalNoiseModel_);
 
         // Start with robust model as default
         activeNoiseModel_ = robustNoiseModel_;
     }
 
     /// Constructor initializes both diagonal and robust models
     SwitchableRobustNoiseModel(const Vector &sigmas, double huberThreshold) : Base(sigmas.size()) {
 
         // Create digonal noise model
         diagonalNoiseModel_ = noiseModel::Diagonal::Sigmas(sigmas);
 
         // Create robust noise model (Huber loss with diagonal base)
         auto robustKernel = noiseModel::mEstimator::Huber::Create(huberThreshold);
         robustNoiseModel_ = noiseModel::Robust::Create(robustKernel, diagonalNoiseModel_);
 
         // Start with robust model as default
         activeNoiseModel_ = robustNoiseModel_;
     }
 
     virtual ~SwitchableRobustNoiseModel() override {}
 
     /// Switch to robust/diagonal model
     void setRobustModelActive(bool val) {
         val ? activeNoiseModel_ = robustNoiseModel_ : activeNoiseModel_ = diagonalNoiseModel_;
     }
 
     /// Get the currently active noise model
     SharedNoiseModel getActiveModel() const { return activeNoiseModel_; }
 
     /// Get the robust noise model
     SharedNoiseModel getRobustModel() const { return robustNoiseModel_; }
 
     /// Get the diagonal noise model
     SharedNoiseModel getDiagonalModel() const { return diagonalNoiseModel_; }
 
   public:
     /// Override `whiten` method to delegate to active model
     virtual Vector whiten(const Vector &v) const override { return activeNoiseModel_->whiten(v); }
 
     /// Override `Whiten` method to delegate to active model
     virtual Matrix Whiten(const Matrix &H) const override { return activeNoiseModel_->Whiten(H); }
 
     /// Override `unwhiten` method to delegate to active model
     virtual Vector unwhiten(const Vector &v) const override {
         return activeNoiseModel_->unwhiten(v);
     }
 
     /// Override `whitenInPlace` method to delegate to active model
     virtual void whitenInPlace(Vector &v) const override { activeNoiseModel_->whitenInPlace(v); }
 
     /// Implement the print method for debugging/inspection
     virtual void print(const std::string &name = "") const override {
         std::cout << name << "SwitchableRobustNoiseModel (currently using: ";
         if (activeNoiseModel_ == diagonalNoiseModel_) {
             std::cout << "Diagonal model";
         } else {
             std::cout << "Robust model";
         }
         activeNoiseModel_->print();
         std::cout << ")\n";
     }
 
     /// Implement the `equals` method for comparison
     virtual bool equals(const noiseModel::Base &expected, double tol = 1e-9) const override {
         const SwitchableRobustNoiseModel *expectedModel =
             dynamic_cast<const SwitchableRobustNoiseModel *>(&expected);
         if (!expectedModel)
             return false;
 
         return activeNoiseModel_->equals(*expectedModel->getActiveModel(), tol);
     }
 
     // Implement the `WhitenSystem` methods as required by the base class
     virtual void WhitenSystem(std::vector<Matrix> &A, Vector &b) const override {
         activeNoiseModel_->WhitenSystem(A, b);
     }
 
     virtual void WhitenSystem(Matrix &A, Vector &b) const override {
         activeNoiseModel_->WhitenSystem(A, b);
     }
 
     virtual void WhitenSystem(Matrix &A1, Matrix &A2, Vector &b) const override {
         activeNoiseModel_->WhitenSystem(A1, A2, b);
     }
 
     virtual void WhitenSystem(Matrix &A1, Matrix &A2, Matrix &A3, Vector &b) const override {
         activeNoiseModel_->WhitenSystem(A1, A2, A3, b);
     }
 };
 
} // namespace gtsam_factors