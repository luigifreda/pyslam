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


//#define GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR  // before including gtsam

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Similarity3.h>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/shared_ptr.hpp>  // Include Boost

#include <iostream>

namespace py = pybind11;

using namespace gtsam;
//using namespace gtsam::noiseModel;
using symbol_shorthand::X;


template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const T& model) {
    return boost::make_shared<T>(model); // This makes a copy of model
}

template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const boost::shared_ptr<T>& model) {
    return model; // No copy, just return the same shared_ptr
}


// =====================================================================================================================


// Generalized numericalDerivative11 template
template <typename Y, typename X>
Eigen::MatrixXd numericalDerivative11General(
    std::function<Y(const X&)> h, const X& x, double delta = 1e-5) {
  
    return numericalDerivative11<Y, X>(h, x, delta);
}

// Specialization for Similarity3 -> Vector2
template <>
Eigen::MatrixXd numericalDerivative11General<Vector2, Similarity3>(
    std::function<Vector2(const Similarity3&)> h, const Similarity3& x, double delta) {
  
    return numericalDerivative11<Vector2, Similarity3>(h, x, delta);
}

// Python wrapper for generalized function
Eigen::MatrixXd numericalDerivative11WrapSim3(
    py::function py_func, const Similarity3& x, double delta = 1e-5) {
    
    std::function<Vector2(const Similarity3&)> func = 
        [py_func](const Similarity3& sim) -> Vector2 {
            return py_func(sim).cast<Vector2>();
        };

    // Generalized numerical derivative function
    return numericalDerivative11General<Vector2, Similarity3>(func, x, delta);
}

// Python wrapper for other types, just in case
template <typename Y, typename X>
Eigen::MatrixXd numericalDerivative11WrapAny(
    py::function py_func, const X& x, double delta = 1e-5) {
    
    std::function<Y(const X&)> func = 
        [py_func](const X& arg) -> Y {
            return py_func(arg).template cast<Y>();
        };

    return numericalDerivative11General<Y, X>(func, x, delta);
}


// =====================================================================================================================



class SwitchableRobustNoiseModel : public noiseModel::Base {
private:
    SharedNoiseModel robustNoiseModel_;
    SharedNoiseModel diagonalNoiseModel_;
    SharedNoiseModel activeNoiseModel_; // Currently active model

public:
    /// Constructor initializes both diagonal/isotropic and robust models
    SwitchableRobustNoiseModel(int dim, double sigma, double huberThreshold):Base(dim) {

        // Create isotropic noise model
        diagonalNoiseModel_ = noiseModel::Isotropic::Sigma(dim, sigma);

        // Create robust noise model (Huber loss with isotropic base)
        auto robustKernel = noiseModel::mEstimator::Huber::Create(huberThreshold);
        robustNoiseModel_ = noiseModel::Robust::Create(robustKernel, diagonalNoiseModel_);

        // Start with robust model as default
        activeNoiseModel_ = robustNoiseModel_;
    }

    /// Constructor initializes both diagonal and robust models
    SwitchableRobustNoiseModel(const Vector& sigmas, double huberThreshold):Base(sigmas.size()) {

        // Create digonal noise model
        diagonalNoiseModel_ = noiseModel::Diagonal::Sigmas(sigmas);

        // Create robust noise model (Huber loss with diagonal base)
        auto robustKernel = noiseModel::mEstimator::Huber::Create(huberThreshold);
        robustNoiseModel_ = noiseModel::Robust::Create(robustKernel, diagonalNoiseModel_);

        // Start with robust model as default
        activeNoiseModel_ = robustNoiseModel_;
    }

    /// Switch to robust/diagonal model
    void setRobustModelActive(bool val) { val? activeNoiseModel_ = robustNoiseModel_ : activeNoiseModel_ = diagonalNoiseModel_; }

    /// Get the currently active noise model
    SharedNoiseModel getActiveModel() const { return activeNoiseModel_; }

    /// Get the robust noise model
    SharedNoiseModel getRobustModel() const { return robustNoiseModel_; }

    /// Get the diagonal noise model
    SharedNoiseModel getDiagonalModel() const { return diagonalNoiseModel_; }

public: 

    /// Override `whiten` method to delegate to active model
    virtual Vector whiten(const Vector& v) const override {
        return activeNoiseModel_->whiten(v);
    }

    /// Override `Whiten` method to delegate to active model
    virtual Matrix Whiten(const Matrix& H) const override {
        return activeNoiseModel_->Whiten(H);
    }

    /// Override `unwhiten` method to delegate to active model
    virtual Vector unwhiten(const Vector& v) const override {
        return activeNoiseModel_->unwhiten(v);
    }

    /// Override `whitenInPlace` method to delegate to active model
    virtual void whitenInPlace(Vector& v) const override {
        activeNoiseModel_->whitenInPlace(v);
    }

    /// Implement the print method for debugging/inspection
    virtual void print(const std::string& name = "") const override {
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
    virtual bool equals(const noiseModel::Base& expected, double tol = 1e-9) const override {
        const SwitchableRobustNoiseModel* expectedModel = dynamic_cast<const SwitchableRobustNoiseModel*>(&expected);
        if (!expectedModel)
            return false;

        return activeNoiseModel_->equals(*expectedModel->getActiveModel(), tol);
    }

    // Implement the `WhitenSystem` methods as required by the base class
    virtual void WhitenSystem(std::vector<Matrix>& A, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A, b);
    }

    virtual void WhitenSystem(Matrix& A, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A, b);
    }

    virtual void WhitenSystem(Matrix& A1, Matrix& A2, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A1, A2, b);
    }

    virtual void WhitenSystem(Matrix& A1, Matrix& A2, Matrix& A3, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A1, A2, A3, b);
    }
};


// =====================================================================================================================


/**
 * Monocular Resectioning Factor
 */
class ResectioningFactor : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    const Cal3_S2& K_;
    Point3 P_;
    Point2 p_;
    double weight_ = 1.0;

    ResectioningFactor(const SharedNoiseModel& model, 
                       const Key& key,
                       const Cal3_S2& calib,
                       const Point2& measured_p,
                       const Point3& world_P)
        : Base(model, key), K_(calib), P_(world_P), p_(measured_p) {}

    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H = boost::none) const override {
        PinholeCamera<Cal3_S2> camera(pose, K_);        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(2,6);
                return Vector::Zero(2);
            } else {
                Vector error = weight_ * (camera.project(P_, H, boost::none, boost::none) - p_);
                if (H) *H *= weight_;
                return error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactor]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(2);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 

/**
 * Stereo Resectioning Factor
 */
class ResectioningFactorStereo : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    Cal3_S2Stereo::shared_ptr K_;    
    Point3 P_;
    StereoPoint2 p_stereo_;
    double weight_ = 1.0;    

    ResectioningFactorStereo(const SharedNoiseModel& model, 
                             const Key& key,
                             const Cal3_S2Stereo& calib,
                             const StereoPoint2& measured_p_stereo,
                             const Point3& world_P)
        : Base(model, key), /*K_(calib),*/ P_(world_P), p_stereo_(measured_p_stereo) {
            K_ = boost::make_shared<Cal3_S2Stereo>(calib);
        }

    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H = boost::none) const override {
        StereoCamera camera(pose, K_);
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(3,6);
                return Vector::Zero(3);
            } else {
                StereoPoint2 projected = camera.project(P_, H, boost::none, boost::none);
                Vector error = weight_ * (Vector(3) << projected.uL() - p_stereo_.uL(),
                                                       projected.uR() - p_stereo_.uR(),
                                                       projected.v() - p_stereo_.v()).finished();
                if (H) *H *= weight_;
                return error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(3,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorStereo]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(3);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }    
};



// =====================================================================================================================



  /**
   * Non-linear factor for a constraint derived from a 2D measurement. 
   * A weight can be applied to the error and Jacobians in order to disable the factor or to make it less important.
   */
  template<class POSE, class LANDMARK, class CALIBRATION = Cal3_S2>
  class WeightedGenericProjectionFactor: public NoiseModelFactor2<POSE, LANDMARK> {
  protected:

    // Keep a copy of measurement and calibration for I/O
    Point2 measured_;                    ///< 2D measurement
    boost::shared_ptr<CALIBRATION> K_;  ///< shared pointer to calibration object
    boost::optional<POSE> body_P_sensor_; ///< The pose of the sensor in the body frame

    // verbosity handling for Cheirality Exceptions
    bool throwCheirality_; ///< If true, rethrows Cheirality exceptions (default: false)
    bool verboseCheirality_; ///< If true, prints text for Cheirality exceptions (default: false)

    double weight_ = 1.0; ///< Positive weighting factor for the error and Jacobians
  public:

    /// shorthand for base class type
    typedef NoiseModelFactor2<POSE, LANDMARK> Base;

    /// shorthand for this class
    typedef WeightedGenericProjectionFactor<POSE, LANDMARK, CALIBRATION> This;

    /// shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<This> shared_ptr;

    /// Default constructor
  WeightedGenericProjectionFactor() :
      measured_(0, 0), throwCheirality_(false), verboseCheirality_(false) {
  }

    /**
     * Constructor
     * TODO: Mark argument order standard (keys, measurement, parameters)
     * @param measured is the 2 dimensional location of point in image (the measurement)
     * @param model is the standard deviation
     * @param poseKey is the index of the camera
     * @param pointKey is the index of the landmark
     * @param K shared pointer to the constant calibration
     * @param body_P_sensor is the transform from body to sensor frame (default identity)
     */
    WeightedGenericProjectionFactor(const Point2& measured, const SharedNoiseModel& model,
        Key poseKey, Key pointKey, const boost::shared_ptr<CALIBRATION>& K,
        boost::optional<POSE> body_P_sensor = boost::none) :
          Base(model, poseKey, pointKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
          throwCheirality_(false), verboseCheirality_(false) {}

    /**
     * Constructor with exception-handling flags
     * TODO: Mark argument order standard (keys, measurement, parameters)
     * @param measured is the 2 dimensional location of point in image (the measurement)
     * @param model is the standard deviation
     * @param poseKey is the index of the camera
     * @param pointKey is the index of the landmark
     * @param K shared pointer to the constant calibration
     * @param throwCheirality determines whether Cheirality exceptions are rethrown
     * @param verboseCheirality determines whether exceptions are printed for Cheirality
     * @param body_P_sensor is the transform from body to sensor frame  (default identity)
     */
    WeightedGenericProjectionFactor(const Point2& measured, const SharedNoiseModel& model,
        Key poseKey, Key pointKey, const boost::shared_ptr<CALIBRATION>& K,
        bool throwCheirality, bool verboseCheirality,
        boost::optional<POSE> body_P_sensor = boost::none) :
          Base(model, poseKey, pointKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
          throwCheirality_(throwCheirality), verboseCheirality_(verboseCheirality) {}

    /** Virtual destructor */
    virtual ~WeightedGenericProjectionFactor() {}

    /// @return a deep copy of this factor
    virtual gtsam::NonlinearFactor::shared_ptr clone() const {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this))); }

    /**
     * print
     * @param s optional string naming the factor
     * @param keyFormatter optional formatter useful for printing Symbols
     */
    void print(const std::string& s = "", const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
      std::cout << s << "WeightedGenericProjectionFactor, z = ";
      traits<Point2>::Print(measured_);
      if(this->body_P_sensor_)
        this->body_P_sensor_->print("  sensor pose in body frame: ");
      Base::print("", keyFormatter);
    }

    /// equals
    virtual bool equals(const NonlinearFactor& p, double tol = 1e-9) const {
      const This *e = dynamic_cast<const This*>(&p);
      return e
          && Base::equals(p, tol)
          && traits<Point2>::Equals(this->measured_, e->measured_, tol)
          && this->K_->equals(*e->K_, tol)
          && std::abs(this->weight_ - e->weight_) < tol
          && ((!body_P_sensor_ && !e->body_P_sensor_) || (body_P_sensor_ && e->body_P_sensor_ && body_P_sensor_->equals(*e->body_P_sensor_)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    Vector evaluateError(const Pose3& pose, const Point3& point,
        boost::optional<Matrix&> H1 = boost::none, boost::optional<Matrix&> H2 = boost::none) const {
      try {
        if (weight_ <= std::numeric_limits<double>::epsilon()) { 
            if (H1) *H1 = Matrix::Zero(2, 6);
            if (H2) *H2 = Matrix::Zero(2, 3);
            return Vector2::Zero();
        }

        if(body_P_sensor_) {
          if(H1) {
            gtsam::Matrix H0;
            PinholeCamera<CALIBRATION> camera(pose.compose(*body_P_sensor_, H0), *K_);
            Point2 reprojectionError(camera.project(point, H1, H2, boost::none) - measured_);
            *H1 = *H1 * H0;
            *H1 *= weight_;
            return weight_ * reprojectionError;
          } else {
            PinholeCamera<CALIBRATION> camera(pose.compose(*body_P_sensor_), *K_);
            Point2 reprojectionError(camera.project(point, H1, H2, boost::none) - measured_);
            return weight_ * reprojectionError;            
          }
        } else {
          PinholeCamera<CALIBRATION> camera(pose, *K_);
          Point2 reprojectionError(camera.project(point, H1, H2, boost::none) - measured_);
          if (H1) *H1 *= weight_;
          if (H2) *H2 *= weight_;
          return weight_ * reprojectionError;             
        }
      } catch( CheiralityException& e) {
        if (H1) *H1 = Matrix::Zero(2,6);
        if (H2) *H2 = Matrix::Zero(2,3);
        if (verboseCheirality_)
          std::cout << e.what() << ": Landmark "<< DefaultKeyFormatter(this->key2()) <<
              " moved behind camera " << DefaultKeyFormatter(this->key1()) << std::endl;
        if (throwCheirality_)
          throw CheiralityException(this->key2());
      }
      return Vector2::Constant(2.0 * K_->fx());
    }

    /** set the weight */
    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }

    /** return the weight */
    double getWeight() const { return weight_; }

    /** return the measurement */
    const Point2& measured() const {
      return measured_;
    }

    /** return the calibration object */
    inline const boost::shared_ptr<CALIBRATION> calibration() const {
      return K_;
    }

    /** return verbosity */
    inline bool verboseCheirality() const { return verboseCheirality_; }

    /** return flag for throwing cheirality exceptions */
    inline bool throwCheirality() const { return throwCheirality_; }

  private:

    /// Serialization function
    friend class boost::serialization::access;
    template<class ARCHIVE>
    void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
      ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
      ar & BOOST_SERIALIZATION_NVP(measured_);
      ar & BOOST_SERIALIZATION_NVP(K_);
      ar & BOOST_SERIALIZATION_NVP(body_P_sensor_);
      ar & BOOST_SERIALIZATION_NVP(throwCheirality_);
      ar & BOOST_SERIALIZATION_NVP(verboseCheirality_);
      ar & BOOST_SERIALIZATION_NVP(weight_);
    }

  public:
    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

using WeightedGenericProjectionFactorCal3_S2 = WeightedGenericProjectionFactor<Pose3, Point3, Cal3_S2> ;
using WeightedGenericProjectionFactorCal3DS2 = WeightedGenericProjectionFactor<Pose3, Point3, Cal3DS2>;



// =====================================================================================================================


/**
 * A Generic Stereo Factor
 * A weight can be applied to the error and Jacobians in order to disable the factor or to make it less important.
 */
template<class POSE=Pose3, class LANDMARK=Point3>
class WeightedGenericStereoFactor: public NoiseModelFactor2<POSE, LANDMARK> {
private:

  // Keep a copy of measurement and calibration for I/O
  StereoPoint2 measured_;                      ///< the measurement
  Cal3_S2Stereo::shared_ptr K_;                ///< shared pointer to calibration
  boost::optional<POSE> body_P_sensor_;        ///< The pose of the sensor in the body frame

  // verbosity handling for Cheirality Exceptions
  bool throwCheirality_;                       ///< If true, rethrows Cheirality exceptions (default: false)
  bool verboseCheirality_;                     ///< If true, prints text for Cheirality exceptions (default: false)

  double weight_ = 1.0;                        ///< Positive weighting factor for the error and Jacobians
public:

  // shorthand for base class type
  typedef NoiseModelFactor2<POSE, LANDMARK> Base;             ///< typedef for base class
  typedef WeightedGenericStereoFactor<POSE, LANDMARK> This;           ///< typedef for this class (with templates)
  typedef boost::shared_ptr<WeightedGenericStereoFactor> shared_ptr;  ///< typedef for shared pointer to this object
  typedef POSE CamPose;                                       ///< typedef for Pose Lie Value type

  /**
   * Default constructor
   */
  WeightedGenericStereoFactor() : K_(new Cal3_S2Stereo(444, 555, 666, 777, 888, 1.0)),
      throwCheirality_(false), verboseCheirality_(false) {}

  /**
   * Constructor
   * @param measured is the Stereo Point measurement (u_l, u_r, v). v will be identical for left & right for rectified stereo pair
   * @param model is the noise model in on the measurement
   * @param poseKey the pose variable key
   * @param landmarkKey the landmark variable key
   * @param K the constant calibration
   * @param body_P_sensor is the transform from body to sensor frame (default identity)
   */
  WeightedGenericStereoFactor(const StereoPoint2& measured, const SharedNoiseModel& model,
      Key poseKey, Key landmarkKey, const Cal3_S2Stereo::shared_ptr& K,
      boost::optional<POSE> body_P_sensor = boost::none) :
    Base(model, poseKey, landmarkKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
    throwCheirality_(false), verboseCheirality_(false) {}

  /**
   * Constructor with exception-handling flags
   * @param measured is the Stereo Point measurement (u_l, u_r, v). v will be identical for left & right for rectified stereo pair
   * @param model is the noise model in on the measurement
   * @param poseKey the pose variable key
   * @param landmarkKey the landmark variable key
   * @param K the constant calibration
   * @param throwCheirality determines whether Cheirality exceptions are rethrown
   * @param verboseCheirality determines whether exceptions are printed for Cheirality
   * @param body_P_sensor is the transform from body to sensor frame  (default identity)
   */
  WeightedGenericStereoFactor(const StereoPoint2& measured, const SharedNoiseModel& model,
      Key poseKey, Key landmarkKey, const Cal3_S2Stereo::shared_ptr& K,
      bool throwCheirality, bool verboseCheirality,
      boost::optional<POSE> body_P_sensor = boost::none) :
    Base(model, poseKey, landmarkKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
    throwCheirality_(throwCheirality), verboseCheirality_(verboseCheirality) {}

  /** Virtual destructor */
  virtual ~WeightedGenericStereoFactor() {}

  /// @return a deep copy of this factor
  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this))); }

  /**
   * print
   * @param s optional string naming the factor
   * @param keyFormatter optional formatter useful for printing Symbols
   */
  void print(const std::string& s = "", const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
    Base::print(s, keyFormatter);
    measured_.print(s + ".z");
    if(this->body_P_sensor_)
      this->body_P_sensor_->print("  sensor pose in body frame: ");
  }

  /**
   * equals
   */
  virtual bool equals(const NonlinearFactor& f, double tol = 1e-9) const {
    const WeightedGenericStereoFactor* e = dynamic_cast<const WeightedGenericStereoFactor*> (&f);
    return e
        && Base::equals(f)
        && measured_.equals(e->measured_, tol)
        && std::abs(this->weight_ - e->weight_) < tol
        && ((!body_P_sensor_ && !e->body_P_sensor_) || (body_P_sensor_ && e->body_P_sensor_ && body_P_sensor_->equals(*e->body_P_sensor_)));
  }

  /** h(x)-z */
  Vector evaluateError(const Pose3& pose, const Point3& point,
      boost::optional<Matrix&> H1 = boost::none, boost::optional<Matrix&> H2 = boost::none) const {
    try {
        if (weight_ <= std::numeric_limits<double>::epsilon()) { 
            if (H1) *H1 = Matrix::Zero(3, 6);
            if (H2) *H2 = Matrix::Zero(3, 3);
            return Vector3::Zero();
     }

      if(body_P_sensor_) {
        if(H1) {
          gtsam::Matrix H0;
          StereoCamera stereoCam(pose.compose(*body_P_sensor_, H0), K_);
          StereoPoint2 reprojectionError(stereoCam.project(point, H1, H2) - measured_);
          *H1 = *H1 * H0;
          *H1 *= weight_;          
          return weight_ * reprojectionError.vector();
        } else {
          StereoCamera stereoCam(pose.compose(*body_P_sensor_), K_);
          StereoPoint2 reprojectionError(stereoCam.project(point, H1, H2) - measured_);
          return weight_ * reprojectionError.vector();
        }
      } else {
        StereoCamera stereoCam(pose, K_);
        StereoPoint2 reprojectionError(stereoCam.project(point, H1, H2) - measured_);
        if (H1) *H1 *= weight_;
        if (H2) *H2 *= weight_;
        return weight_ * reprojectionError.vector();
      }
    } catch(StereoCheiralityException& e) {
      if (H1) *H1 = Matrix::Zero(3,6);
      if (H2) *H2 = Z_3x3;
      if (verboseCheirality_)
      std::cout << e.what() << ": Landmark "<< DefaultKeyFormatter(this->key2()) <<
          " moved behind camera " << DefaultKeyFormatter(this->key1()) << std::endl;
      if (throwCheirality_)
        throw StereoCheiralityException(this->key2());
    }
    return Vector3::Constant(2.0 * K_->fx());
  }


  /** set the weight */
  void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }

  /** return the weight */
  double getWeight() const { return weight_; }

  /** return the measured */
  const StereoPoint2& measured() const {
    return measured_;
  }

  /** return the calibration object */
  inline const Cal3_S2Stereo::shared_ptr calibration() const {
    return K_;
  }

  /** return verbosity */
  inline bool verboseCheirality() const { return verboseCheirality_; }

  /** return flag for throwing cheirality exceptions */
  inline bool throwCheirality() const { return throwCheirality_; }

private:
  /** Serialization function */
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int /*version*/) {
    ar & boost::serialization::make_nvp("NoiseModelFactor2",
        boost::serialization::base_object<Base>(*this));
    ar & BOOST_SERIALIZATION_NVP(measured_);
    ar & BOOST_SERIALIZATION_NVP(K_);
    ar & BOOST_SERIALIZATION_NVP(body_P_sensor_);
    ar & BOOST_SERIALIZATION_NVP(throwCheirality_);
    ar & BOOST_SERIALIZATION_NVP(verboseCheirality_);
    ar & BOOST_SERIALIZATION_NVP(weight_);
  }
};

using WeightedGenericStereoFactor3D = WeightedGenericStereoFactor<gtsam::Pose3, gtsam::Point3>;


// =====================================================================================================================


// Similarity3 prior factor with autodifferencing. Goal is to penalize all terms.
class PriorFactorSimilarity3 : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    gtsam::Similarity3 prior_;

    PriorFactorSimilarity3(gtsam::Key key, const gtsam::Similarity3& prior, const gtsam::SharedNoiseModel& noise)
        : Base(noise, key), prior_(prior) {}

    // Define error function: Logmap of transformation error
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        gtsam::Vector7 error = gtsam::Similarity3::Logmap(prior_.inverse() * sim);

        if (H) {
            auto functor = [this](const gtsam::Similarity3& sim) {
                return gtsam::Similarity3::Logmap(prior_.inverse() * sim);
            };
            *H = gtsam::numericalDerivative11<gtsam::Vector7, gtsam::Similarity3>(functor, sim, 1e-5);
        }
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3>(*this);
    }
};


// Similarity3 prior factor with only scale. Goal is to only penalize scale difference.
class PriorFactorSimilarity3ScaleOnly : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    double prior_scale_;  // Prior scale value

    PriorFactorSimilarity3ScaleOnly(gtsam::Key key, double prior_scale, double prior_sigma)
        : Base(gtsam::noiseModel::Isotropic::Sigma(1, prior_sigma), key), prior_scale_(prior_scale) {}

    // Define error function: Only penalize scale difference
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        double scale_error = std::log(sim.scale()) - std::log(prior_scale_);  // Log-scale difference

        if (H) {
            // Compute Jacobian: d(log(s)) / ds = 1 / s
            Eigen::Matrix<double, 1, 7> J = Eigen::Matrix<double, 1, 7>::Zero();
            J(6) = 1.0 / sim.scale();  // Derivative w.r.t. scale
            *H = J;
        }
        return gtsam::Vector1(scale_error);  // Return 1D error
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3ScaleOnly>(*this);
    }
};


// =====================================================================================================================


// Used by optimizer_gtsam.optimize_sim3()
class SimResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
private:
    const Cal3_S2& calib_;  // Camera intrinsics
    const Point2 uv_;       // Observed 2D point
    const Point3 P_;        // 3D point
    double weight_ = 1.0;   // Weight

public:
    SimResectioningFactor(const gtsam::Key& sim_pose_key,
                            const gtsam::Cal3_S2& calib,
                            const gtsam::Point2& uv,
                            const gtsam::Point3& P,
                            const gtsam::SharedNoiseModel& noiseModel)
        : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
            calib_(calib), uv_(uv), P_(P) {}

    gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        auto computeError = [this](const gtsam::Similarity3& sim) {
            const gtsam::Matrix3 R = sim.rotation().matrix();
            const gtsam::Vector3 t = sim.translation();
            const double s = sim.scale();
            const gtsam::Vector3 transformed_P = s * (R * P_) + t;
            const gtsam::Vector3 projected = calib_.K() * transformed_P;
            const gtsam::Point2 uv = projected.head<2>() / projected[2];
            const gtsam::Vector2 error = uv - uv_;
            return error;
        };

        const gtsam::Vector2 error = weight_ * computeError(sim3);

        // Compute Jacobians if required
        if (H) {    
            *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Similarity3>(computeError, sim3, 1e-5);
        }
        return error;
    }

    void setWeight(double weight) {
        weight_ = weight;
    }

    double getWeight() const {
        return weight_;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<SimResectioningFactor>(*this);
    }    
};


// Used by optimizer_gtsam.optimize_sim3()
class SimInvResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
private:
    const Cal3_S2& calib_;  // Camera intrinsics
    const Point2 uv_;       // Observed 2D pixel point
    const Point3 P_;        // 3D camera point
    double weight_ = 1.0;   // Weight    

public:
    SimInvResectioningFactor(const gtsam::Key& sim_pose_key,
                             const gtsam::Cal3_S2& calib,
                             const gtsam::Point2& p,
                             const gtsam::Point3& P,
                             const gtsam::SharedNoiseModel& noiseModel)
        : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
            calib_(calib), uv_(p), P_(P) {}

    gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        auto computeError = [this](const gtsam::Similarity3& sim) {
            const gtsam::Matrix3 R = sim.rotation().matrix();
            const gtsam::Vector3 t = sim.translation();
            const double s = sim.scale();

            const gtsam::Matrix3 R_inv = R.transpose();
            const double s_inv = 1.0 / s;
            const gtsam::Vector3 t_inv = -s_inv * (R_inv * t);
            const gtsam::Vector3 transformed_P = s_inv * (R_inv * P_) + t_inv;
            const gtsam::Vector3 projected = calib_.K() * transformed_P;
            const gtsam::Point2 uv = projected.head<2>() / projected[2];
            const gtsam::Vector2 error = uv - uv_;
            return error;
        };   

        const gtsam::Vector2 error = weight_ * computeError(sim3);

        // Compute Jacobians if required
        if (H) {
            *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Similarity3>(computeError, sim3, 1e-5);
        }

        return error;
    }

    void setWeight(double weight) {
        weight_ = weight;
    }

    double getWeight() const {
        return weight_;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<SimInvResectioningFactor>(*this);
    }
};
    

// =====================================================================================================================


// Custom version of BetweenFactor<Similarity3> with autodifferencing
class BetweenFactorSimilarity3 : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>;
    gtsam::Similarity3 measured_; // Relative Sim3 measurement

    BetweenFactorSimilarity3(gtsam::Key key1, gtsam::Key key2, const gtsam::Similarity3& measured, const gtsam::SharedNoiseModel& model)
        : Base(model, key1, key2), measured_(measured) {}

    // Compute error (7D residual)
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim3_1, const gtsam::Similarity3& sim3_2,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        // Compute predicted relative transformation
        gtsam::Similarity3 predicted = sim3_1.inverse() * sim3_2; // Swc1.inverse() * Swc2 = Sc1c2 = S12
        
        // Compute the error in Sim3 space
        gtsam::Similarity3 errorSim3 = measured_ * predicted.inverse();
        
        // Compute Jacobians only if needed
        if (H1) {
            *H1 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3& s1) { 
                    return Similarity3::Logmap(measured_ * (s1.inverse() * sim3_2).inverse()); 
                }, sim3_1, 1e-5);
        }
        if (H2) {
            *H2 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3& s2) { 
                    return Similarity3::Logmap(measured_ * (sim3_1.inverse() * s2).inverse()); 
                }, sim3_2, 1e-5);
        }

        // Log map to get minimal 7D error representation
        return Similarity3::Logmap(errorSim3);
    }
};


// =====================================================================================================================


// Function to insert Similarity3 into Values
void insertSimilarity3(gtsam::Values &values, const Key& key, const gtsam::Similarity3 &sim3) {
    values.insert(key, sim3);
}

// Function to get Similarity3 from Values
gtsam::Similarity3 getSimilarity3(const gtsam::Values &values, gtsam::Key key) {
    if (!values.exists(key)) {
        throw std::runtime_error("Key not found in Values.");
    }
    return values.at<gtsam::Similarity3>(key);  // Return by value (safe)
}

// =====================================================================================================================


PYBIND11_MODULE(gtsam_factors, m) {
    py::module_::import("gtsam");  // Ensure GTSAM is loaded


    // Register NoiseModelFactorN classes
    py::class_<NoiseModelFactorN<Pose3>, std::shared_ptr<NoiseModelFactorN<Pose3>>, NonlinearFactor>(m, "NoiseModelFactorN_Pose3");
    py::class_<NoiseModelFactorN<Similarity3>, std::shared_ptr<NoiseModelFactorN<Similarity3>>, NonlinearFactor>(m, "NoiseModelFactorN_Similarity3");

    // Register base class for BetweenFactorSimilarity3
    py::class_<gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>,gtsam::NonlinearFactor, std::shared_ptr<gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>>>(m, "NoiseModelFactor2Similarity3")
    .def("print", &gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>::print);


    // NOTE: This does not work. We need to specialize the template for each type we are interested as it is done for Similarity3 in numerical_derivative11_sim3
    // m.def("numerical_derivative11", 
    //     [](py::function py_func, const auto& x, double delta = 1e-5) -> Eigen::MatrixXd {
    //         // Explicitly define the lambda signature
    //         return numericalDerivative11WrapAny(py_func, x, delta);
    //     },
    //     py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
    //     "Compute numerical derivative of a function mapping any type");

    // Specialization of numericalDerivative11 for Similarity3 -> Vector2
    m.def("numerical_derivative11_sim3", 
        [](py::function py_func, const Similarity3& x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return numericalDerivative11WrapSim3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Similarity3 to Vector2");


    py::class_<SwitchableRobustNoiseModel, std::shared_ptr<SwitchableRobustNoiseModel>, noiseModel::Base>(m, "SwitchableRobustNoiseModel")
    .def(py::init([](int dim, double sigma, double huberThreshold) {
        return new SwitchableRobustNoiseModel(dim, sigma, huberThreshold);
    }), py::arg("dim"), py::arg("sigma"), py::arg("huberThreshold"))
    .def(py::init([](const Vector& sigmas, double huberThreshold) {
        return new SwitchableRobustNoiseModel(sigmas, huberThreshold);
    }), py::arg("sigmas"), py::arg("huberThreshold"))
    .def("set_robust_model_active", &SwitchableRobustNoiseModel::setRobustModelActive)
    .def("get_robust_model", &SwitchableRobustNoiseModel::getRobustModel)
    .def("get_diagonal_model", &SwitchableRobustNoiseModel::getDiagonalModel);

    
    py::class_<ResectioningFactor, std::shared_ptr<ResectioningFactor>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactor")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(model, key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Diagonal& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const SwitchableRobustNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactor::evaluateError)
    .def("error", &ResectioningFactor::error)
    .def("set_weight", &ResectioningFactor::setWeight)
    .def("get_weight", &ResectioningFactor::getWeight);


    py::class_<ResectioningFactorStereo, std::shared_ptr<ResectioningFactorStereo>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactorStereo")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(model, key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Diagonal& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const SwitchableRobustNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactorStereo::evaluateError)
    .def("error", &ResectioningFactorStereo::error)
    .def("set_weight", &ResectioningFactorStereo::setWeight)
    .def("get_weight", &ResectioningFactorStereo::getWeight);    


    py::class_<WeightedGenericProjectionFactorCal3_S2, gtsam::NonlinearFactor, std::shared_ptr<WeightedGenericProjectionFactorCal3_S2>>(m, "WeightedGenericProjectionFactorCal3_S2")
    .def(py::init([] (const Point2& measured, const SharedNoiseModel& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
            return new WeightedGenericProjectionFactorCal3_S2(measured, model, poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))
    .def(py::init([] (const Point2& measured, const noiseModel::Diagonal& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
        return new WeightedGenericProjectionFactorCal3_S2(measured, create_shared_noise_model(model), poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))   
    .def(py::init([] (const Point2& measured, const noiseModel::Robust& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
        return new WeightedGenericProjectionFactorCal3_S2(measured, create_shared_noise_model(model), poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))    
    .def(py::init([] (const Point2& measured, const SwitchableRobustNoiseModel& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
        return new WeightedGenericProjectionFactorCal3_S2(measured, create_shared_noise_model(model), poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))         
    .def("get_weight", &WeightedGenericProjectionFactorCal3_S2::getWeight)
    .def("set_weight", &WeightedGenericProjectionFactorCal3_S2::setWeight);


    py::class_<WeightedGenericStereoFactor3D, gtsam::NonlinearFactor, std::shared_ptr<WeightedGenericStereoFactor3D>>(m, "WeightedGenericStereoFactor")
    .def(py::init([] (const StereoPoint2& measured, const SharedNoiseModel& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
            return new WeightedGenericStereoFactor3D(measured, model, poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K")) 
    .def(py::init([] (const StereoPoint2& measured, const noiseModel::Diagonal& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
        return new WeightedGenericStereoFactor3D(measured, create_shared_noise_model(model), poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K"))
    .def(py::init([] (const StereoPoint2& measured, const noiseModel::Robust& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
        return new WeightedGenericStereoFactor3D(measured, create_shared_noise_model(model), poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K")) 
    .def(py::init([] (const StereoPoint2& measured, const SwitchableRobustNoiseModel& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
        return new WeightedGenericStereoFactor3D(measured, create_shared_noise_model(model), poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K"))              
    .def("get_weight", &WeightedGenericStereoFactor3D::getWeight)
    .def("set_weight", &WeightedGenericStereoFactor3D::setWeight);


    py::class_<PriorFactorSimilarity3, gtsam::NoiseModelFactor1<gtsam::Similarity3>, std::shared_ptr<PriorFactorSimilarity3>>(m, "PriorFactorSimilarity3")
    .def(py::init<const gtsam::Key&, const gtsam::Similarity3&, const gtsam::SharedNoiseModel&>(),
        py::arg("key"), py::arg("prior"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key, const gtsam::Similarity3& prior, const gtsam::noiseModel::Isotropic& model){
        return new PriorFactorSimilarity3(key, prior, create_shared_noise_model(model));       
    }), py::arg("key"), py::arg("prior"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key, const gtsam::Similarity3& prior, const gtsam::noiseModel::Diagonal& model){
        return new PriorFactorSimilarity3(key, prior, create_shared_noise_model(model));
    }), py::arg("key"), py::arg("prior"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key, const gtsam::Similarity3& prior, const gtsam::noiseModel::Robust& model){
        return new PriorFactorSimilarity3(key, prior, create_shared_noise_model(model));       
    }), py::arg("key"), py::arg("prior"), py::arg("model"))
    .def("evaluateError", &PriorFactorSimilarity3::evaluateError);


    py::class_<PriorFactorSimilarity3ScaleOnly, gtsam::NoiseModelFactor1<gtsam::Similarity3>, std::shared_ptr<PriorFactorSimilarity3ScaleOnly>>(m, "PriorFactorSimilarity3ScaleOnly")
    .def(py::init<const gtsam::Key&, double, double>(),
        py::arg("key"), py::arg("prior_scale"), py::arg("prior_sigma"))
    .def("evaluateError", &PriorFactorSimilarity3ScaleOnly::evaluateError);


    py::class_<SimResectioningFactor, gtsam::NoiseModelFactor, std::shared_ptr<SimResectioningFactor>>(m, "SimResectioningFactor")
    .def(py::init<const gtsam::Key&, const gtsam::Cal3_S2&, const gtsam::Point2&, const gtsam::Point3&, const gtsam::SharedNoiseModel&>(),
            py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("noise_model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Diagonal& model){
        return new SimResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Robust& model){
        return new SimResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def("evaluateError", &SimResectioningFactor::evaluateError)
    .def("get_weight", &SimResectioningFactor::getWeight)
    .def("set_weight", &SimResectioningFactor::setWeight);


    py::class_<SimInvResectioningFactor, gtsam::NoiseModelFactor, std::shared_ptr<SimInvResectioningFactor>>(m, "SimInvResectioningFactor")
    .def(py::init<const gtsam::Key&, const gtsam::Cal3_S2&, const gtsam::Point2&, const gtsam::Point3&, const gtsam::SharedNoiseModel&>(),
            py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("noise_model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Diagonal& model){
        return new SimInvResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Robust& model){
        return new SimInvResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def("evaluateError", &SimInvResectioningFactor::evaluateError)
    .def("get_weight", &SimInvResectioningFactor::getWeight)
    .def("set_weight", &SimInvResectioningFactor::setWeight);

    
    m.def("insert_similarity3", &insertSimilarity3, "Insert Similarity3 into Values",
        py::arg("values"), py::arg("key"), py::arg("sim3"));    

    m.def("get_similarity3", &getSimilarity3, "Get Similarity3 from Values",
        py::arg("values"), py::arg("key"));

#if 0
    // Expose BetweenFactor with Similarity3
    py::class_<BetweenFactor<Similarity3>, gtsam::NonlinearFactor, std::shared_ptr<BetweenFactor<Similarity3>>>(m, "BetweenFactorSimilarity3")
    .def(py::init<const Key&, const Key&, const Similarity3&>(), 
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"))
    .def(py::init<const Key&, const Key&, const Similarity3&, const SharedNoiseModel&>(),
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
    .def(py::init([](const Key& key1, const Key& key2, const Similarity3& similarity3, const noiseModel::Diagonal& model) {
        return std::make_shared<BetweenFactor<Similarity3>>(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def(py::init([](const Key& key1, const Key& key2, const Similarity3& similarity3, const noiseModel::Robust& model) {
        return std::make_shared<BetweenFactor<Similarity3>>(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def("evaluateError", &BetweenFactor<Similarity3>::evaluateError)
    .def("print", &BetweenFactor<Similarity3>::print)
    .def("measured", &BetweenFactor<Similarity3>::measured)
    .def("__repr__", [](const BetweenFactor<Similarity3>& self) {
        std::ostringstream ss;
        self.print(ss.str());
        return ss.str();
    });
#else 
    py::class_<BetweenFactorSimilarity3, gtsam::NoiseModelFactor2<Similarity3, Similarity3>, std::shared_ptr<BetweenFactorSimilarity3>>(m, "BetweenFactorSimilarity3")
    .def(py::init<const gtsam::Key&, const gtsam::Key&, const gtsam::Similarity3&, const gtsam::SharedNoiseModel&>(),
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
    .def(py::init([](const gtsam::Key& key1, const gtsam::Key& key2, const gtsam::Similarity3& similarity3, const gtsam::noiseModel::Diagonal& model) {
        return new BetweenFactorSimilarity3(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key1, const gtsam::Key& key2, const gtsam::Similarity3& similarity3, const gtsam::noiseModel::Robust& model) {
        return new BetweenFactorSimilarity3(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def("evaluateError", &BetweenFactorSimilarity3::evaluateError)
    .def("print", &BetweenFactorSimilarity3::print)
    .def("__repr__", [](const BetweenFactorSimilarity3& self) {
        std::ostringstream ss;
        self.print(ss.str());
        return ss.str();    
    });
#endif

}