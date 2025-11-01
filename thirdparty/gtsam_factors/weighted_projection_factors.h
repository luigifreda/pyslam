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


#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>

#include <iostream>
#include <boost/shared_ptr.hpp>  // Include Boost

using namespace gtsam;
using symbol_shorthand::X;

namespace gtsam_factors {

/**
 * Non-linear factor for a constraint derived from a 2D measurement.
 * A weight can be applied to the error and Jacobians in order to disable the factor or to make it less important.
 */
template <class POSE, class LANDMARK, class CALIBRATION = Cal3_S2>
class WeightedGenericProjectionFactor : public NoiseModelFactor2<POSE, LANDMARK>
{
protected:
    // Keep a copy of measurement and calibration for I/O
    Point2 measured_;                     ///< 2D measurement
    boost::shared_ptr<CALIBRATION> K_;    ///< shared pointer to calibration object
    boost::optional<POSE> body_P_sensor_; ///< The pose of the sensor in the body frame

    // verbosity handling for Cheirality Exceptions
    bool throwCheirality_;   ///< If true, rethrows Cheirality exceptions (default: false)
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
    WeightedGenericProjectionFactor() : measured_(0, 0), throwCheirality_(false), verboseCheirality_(false)
    {
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
    WeightedGenericProjectionFactor(const Point2 &measured, const SharedNoiseModel &model,
                                    Key poseKey, Key pointKey, const boost::shared_ptr<CALIBRATION> &K,
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
    WeightedGenericProjectionFactor(const Point2 &measured, const SharedNoiseModel &model,
                                    Key poseKey, Key pointKey, const boost::shared_ptr<CALIBRATION> &K,
                                    bool throwCheirality, bool verboseCheirality,
                                    boost::optional<POSE> body_P_sensor = boost::none) : 
                                    Base(model, poseKey, pointKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
                                        throwCheirality_(throwCheirality), verboseCheirality_(verboseCheirality) {}

    /** Virtual destructor */
    virtual ~WeightedGenericProjectionFactor() {}

    /// @return a deep copy of this factor
    virtual gtsam::NonlinearFactor::shared_ptr clone() const
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /**
     * print
     * @param s optional string naming the factor
     * @param keyFormatter optional formatter useful for printing Symbols
     */
    void print(const std::string &s = "", const KeyFormatter &keyFormatter = DefaultKeyFormatter) const
    {
        std::cout << s << "WeightedGenericProjectionFactor, z = ";
        traits<Point2>::Print(measured_);
        if (this->body_P_sensor_)
            this->body_P_sensor_->print("  sensor pose in body frame: ");
        Base::print("", keyFormatter);
    }

    /// equals
    virtual bool equals(const NonlinearFactor &p, double tol = 1e-9) const
    {
        const This *e = dynamic_cast<const This *>(&p);
        return e && Base::equals(p, tol) && traits<Point2>::Equals(this->measured_, e->measured_, tol) && this->K_->equals(*e->K_, tol) && std::abs(this->weight_ - e->weight_) < tol && ((!body_P_sensor_ && !e->body_P_sensor_) || (body_P_sensor_ && e->body_P_sensor_ && body_P_sensor_->equals(*e->body_P_sensor_)));
    }

    /// Evaluate error h(x)-z and optionally derivatives
    Vector evaluateError(const Pose3 &pose, const Point3 &point,
                         boost::optional<Matrix &> H1 = boost::none, boost::optional<Matrix &> H2 = boost::none) const
    {
        try
        {
            if (weight_ <= std::numeric_limits<double>::epsilon())
            {
                if (H1)
                    *H1 = Matrix::Zero(2, 6);
                if (H2)
                    *H2 = Matrix::Zero(2, 3);
                return Vector2::Zero();
            }

            if (body_P_sensor_)
            {
                if (H1)
                {
                    gtsam::Matrix H0;
                    PinholeCamera<CALIBRATION> camera(pose.compose(*body_P_sensor_, H0), *K_);
                    Point2 reprojectionError(camera.project(point, H1, H2, boost::none) - measured_);
                    *H1 = *H1 * H0;
                    *H1 *= weight_;
                    return weight_ * reprojectionError;
                }
                else
                {
                    PinholeCamera<CALIBRATION> camera(pose.compose(*body_P_sensor_), *K_);
                    Point2 reprojectionError(camera.project(point, H1, H2, boost::none) - measured_);
                    return weight_ * reprojectionError;
                }
            }
            else
            {
                PinholeCamera<CALIBRATION> camera(pose, *K_);
                Point2 reprojectionError(camera.project(point, H1, H2, boost::none) - measured_);
                if (H1)
                    *H1 *= weight_;
                if (H2)
                    *H2 *= weight_;
                return weight_ * reprojectionError;
            }
        }
        catch (CheiralityException &e)
        {
            if (H1)
                *H1 = Matrix::Zero(2, 6);
            if (H2)
                *H2 = Matrix::Zero(2, 3);
            if (verboseCheirality_)
                std::cout << e.what() << ": Landmark " << DefaultKeyFormatter(this->key2()) << " moved behind camera " << DefaultKeyFormatter(this->key1()) << std::endl;
            if (throwCheirality_)
                throw CheiralityException(this->key2());
        }
        return Vector2::Constant(2.0 * K_->fx());
    }

    /** set the weight */
    void setWeight(double weight)
    {
        assert(weight > 0.0);
        weight_ = weight;
    }

    /** return the weight */
    double getWeight() const { return weight_; }

    /** return the measurement */
    const Point2 &measured() const
    {
        return measured_;
    }

    /** return the calibration object */
    inline const boost::shared_ptr<CALIBRATION> calibration() const
    {
        return K_;
    }

    /** return verbosity */
    inline bool verboseCheirality() const { return verboseCheirality_; }

    /** return flag for throwing cheirality exceptions */
    inline bool throwCheirality() const { return throwCheirality_; }

private:
    /// Serialization function
    friend class boost::serialization::access;
    template <class ARCHIVE>
    void serialize(ARCHIVE &ar, const unsigned int /*version*/)
    {
        ar &BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
        ar &BOOST_SERIALIZATION_NVP(measured_);
        ar &BOOST_SERIALIZATION_NVP(K_);
        ar &BOOST_SERIALIZATION_NVP(body_P_sensor_);
        ar &BOOST_SERIALIZATION_NVP(throwCheirality_);
        ar &BOOST_SERIALIZATION_NVP(verboseCheirality_);
        ar &BOOST_SERIALIZATION_NVP(weight_);
    }

public:
    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

using WeightedGenericProjectionFactorCal3_S2 = WeightedGenericProjectionFactor<Pose3, Point3, Cal3_S2>;
using WeightedGenericProjectionFactorCal3DS2 = WeightedGenericProjectionFactor<Pose3, Point3, Cal3DS2>;

// =====================================================================================================================

/**
 * A Generic Stereo Factor
 * A weight can be applied to the error and Jacobians in order to disable the factor or to make it less important.
 */
template <class POSE, class LANDMARK, class CALIBRATION = Cal3_S2Stereo>
class WeightedGenericStereoProjectionFactor : public NoiseModelFactor2<POSE, LANDMARK>
{
private:
    // Keep a copy of measurement and calibration for I/O
    StereoPoint2 measured_;               ///< the measurement
    boost::shared_ptr<CALIBRATION> K_;    ///< shared pointer to calibration
    boost::optional<POSE> body_P_sensor_; ///< The pose of the sensor in the body frame

    // verbosity handling for Cheirality Exceptions
    bool throwCheirality_;   ///< If true, rethrows Cheirality exceptions (default: false)
    bool verboseCheirality_; ///< If true, prints text for Cheirality exceptions (default: false)

    double weight_ = 1.0; ///< Positive weighting factor for the error and Jacobians
public:
    // shorthand for base class type
    typedef NoiseModelFactor2<POSE, LANDMARK> Base;                   

    /// shorthand for this class
    typedef WeightedGenericStereoProjectionFactor<POSE, LANDMARK, CALIBRATION> This;          

    /// shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<This> shared_ptr; 

    /// shorthand for Pose Lie Value type
    typedef POSE CamPose;                       

    /**
     * Default constructor
     */
    WeightedGenericStereoProjectionFactor() :
        measured_(0, 0, 0),
        throwCheirality_(false),
        verboseCheirality_(false)
    {
    }

    /**
     * Constructor
     * @param measured is the Stereo Point measurement (u_l, u_r, v). v will be identical for left & right for rectified stereo pair
     * @param model is the noise model in on the measurement
     * @param poseKey the pose variable key
     * @param pointKey the landmark variable key
     * @param K the constant calibration
     * @param body_P_sensor is the transform from body to sensor frame (default identity)
     */
    WeightedGenericStereoProjectionFactor(const StereoPoint2 &measured, const SharedNoiseModel &model,
                                Key poseKey, Key pointKey, const boost::shared_ptr<CALIBRATION> &K,
                                boost::optional<POSE> body_P_sensor = boost::none) : 
                                Base(model, poseKey, pointKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
                                    throwCheirality_(false), verboseCheirality_(false) {}

    /**
     * Constructor with exception-handling flags
     * @param measured is the Stereo Point measurement (u_l, u_r, v). v will be identical for left & right for rectified stereo pair
     * @param model is the noise model in on the measurement
     * @param poseKey the pose variable key
     * @param pointKey the landmark variable key
     * @param K the constant calibration
     * @param throwCheirality determines whether Cheirality exceptions are rethrown
     * @param verboseCheirality determines whether exceptions are printed for Cheirality
     * @param body_P_sensor is the transform from body to sensor frame  (default identity)
     */
    WeightedGenericStereoProjectionFactor(const StereoPoint2 &measured, const SharedNoiseModel &model,
                                Key poseKey, Key pointKey, const boost::shared_ptr<CALIBRATION> &K,
                                bool throwCheirality, bool verboseCheirality,
                                boost::optional<POSE> body_P_sensor = boost::none) : 
                                Base(model, poseKey, pointKey), measured_(measured), K_(K), body_P_sensor_(body_P_sensor),
                                    throwCheirality_(throwCheirality), verboseCheirality_(verboseCheirality) {}

    /** Virtual destructor */
    virtual ~WeightedGenericStereoProjectionFactor() {}

    /// @return a deep copy of this factor
    virtual gtsam::NonlinearFactor::shared_ptr clone() const
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /**
     * print
     * @param s optional string naming the factor
     * @param keyFormatter optional formatter useful for printing Symbols
     */
    void print(const std::string &s = "", const KeyFormatter &keyFormatter = DefaultKeyFormatter) const
    {
        Base::print(s, keyFormatter);
        measured_.print(s + ".z");
        if (this->body_P_sensor_)
            this->body_P_sensor_->print("  sensor pose in body frame: ");
    }

    /**
     * equals
     */
    virtual bool equals(const NonlinearFactor &f, double tol = 1e-9) const
    {
        const This *e = dynamic_cast<const This *>(&f);
        return e && Base::equals(f, tol) && measured_.equals(e->measured_, tol) && std::abs(this->weight_ - e->weight_) < tol && ((!body_P_sensor_ && !e->body_P_sensor_) || (body_P_sensor_ && e->body_P_sensor_ && body_P_sensor_->equals(*e->body_P_sensor_)));
    }

    /** h(x)-z */
    Vector evaluateError(const Pose3 &pose, const Point3 &point,
                         boost::optional<Matrix &> H1 = boost::none, boost::optional<Matrix &> H2 = boost::none) const
    {
        try
        {
            if (weight_ <= std::numeric_limits<double>::epsilon())
            {
                if (H1)
                    *H1 = Matrix::Zero(3, 6);
                if (H2)
                    *H2 = Matrix::Zero(3, 3);
                return Vector3::Zero();
            }

            if (body_P_sensor_)
            {
                if (H1)
                {
                    gtsam::Matrix H0;
                    StereoCamera stereoCam(pose.compose(*body_P_sensor_, H0), K_);
                    StereoPoint2 reprojectionError(stereoCam.project(point, H1, H2) - measured_);
                    *H1 = *H1 * H0;
                    *H1 *= weight_;
                    return weight_ * reprojectionError.vector();
                }
                else
                {
                    StereoCamera stereoCam(pose.compose(*body_P_sensor_), K_);
                    StereoPoint2 reprojectionError(stereoCam.project(point, H1, H2) - measured_);
                    return weight_ * reprojectionError.vector();
                }
            }
            else
            {
                StereoCamera stereoCam(pose, K_);
                StereoPoint2 reprojectionError(stereoCam.project(point, H1, H2) - measured_);
                if (H1)
                    *H1 *= weight_;
                if (H2)
                    *H2 *= weight_;
                return weight_ * reprojectionError.vector();
            }
        }
        catch (StereoCheiralityException &e)
        {
            if (H1)
                *H1 = Matrix::Zero(3, 6);
            if (H2)
                *H2 = Matrix::Zero(3, 3);
            if (verboseCheirality_)
                std::cout << e.what() << ": Landmark " << DefaultKeyFormatter(this->key2()) << " moved behind camera " << DefaultKeyFormatter(this->key1()) << std::endl;
            if (throwCheirality_)
                throw StereoCheiralityException(this->key2());
        }
        return Vector3::Constant(2.0 * K_->fx());
    }

    /** set the weight */
    void setWeight(double weight)
    {
        assert(weight > 0.0);
        weight_ = weight;
    }

    /** return the weight */
    double getWeight() const { return weight_; }

    /** return the measured */
    const StereoPoint2 &measured() const
    {
        return measured_;
    }

    /** return the calibration object */
    inline const boost::shared_ptr<CALIBRATION> calibration() const
    {
        return K_;
    }

    /** return verbosity */
    inline bool verboseCheirality() const { return verboseCheirality_; }

    /** return flag for throwing cheirality exceptions */
    inline bool throwCheirality() const { return throwCheirality_; }

private:
    /** Serialization function */
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
        ar &boost::serialization::make_nvp("NoiseModelFactor2",
                                           boost::serialization::base_object<Base>(*this));
        ar &BOOST_SERIALIZATION_NVP(measured_);
        ar &BOOST_SERIALIZATION_NVP(K_);
        ar &BOOST_SERIALIZATION_NVP(body_P_sensor_);
        ar &BOOST_SERIALIZATION_NVP(throwCheirality_);
        ar &BOOST_SERIALIZATION_NVP(verboseCheirality_);
        ar &BOOST_SERIALIZATION_NVP(weight_);
    }
};

using WeightedGenericStereoProjectionFactor3D = WeightedGenericStereoProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2Stereo>;

} // namespace gtsam_factors
