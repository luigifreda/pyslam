
#ifndef RxSO3_HEADER
#define RxSO3_HEADER

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include "common.h"

template <typename Scalar>
class RxSO3 {
  public:
    const static int constexpr K = 4; // manifold dimension
    const static int constexpr N = 5; // embedding dimension

    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector4 = Eigen::Matrix<Scalar,4,1>;
    using Matrix3 = Eigen::Matrix<Scalar,3,3>;

    using Tangent = Eigen::Matrix<Scalar,K,1>;
    using Data = Eigen::Matrix<Scalar,N,1>;

    using Point = Eigen::Matrix<Scalar,3,1>;
    using Point4 = Eigen::Matrix<Scalar,4,1>;

    using Quaternion = Eigen::Quaternion<Scalar>;
    using Transformation = Eigen::Matrix<Scalar,3,3>;
    using Adjoint = Eigen::Matrix<Scalar,4,4>;


    EIGEN_DEVICE_FUNC RxSO3(Quaternion const& q, Scalar const s) 
        : unit_quaternion(q), scale(s) {
      unit_quaternion.normalize();
    };

    EIGEN_DEVICE_FUNC RxSO3(const Scalar *data) : unit_quaternion(data), scale(data[4]) {
      unit_quaternion.normalize();
    };

    EIGEN_DEVICE_FUNC RxSO3() {
      unit_quaternion = Quaternion::Identity();
      scale = Scalar(1.0);
    }

    EIGEN_DEVICE_FUNC RxSO3<Scalar> inv() {
      return RxSO3<Scalar>(unit_quaternion.conjugate(), 1.0/scale);
    }

    EIGEN_DEVICE_FUNC Data data() const {
      Data data_vec; data_vec << unit_quaternion.coeffs(), scale;
      return data_vec;
    }

    EIGEN_DEVICE_FUNC RxSO3<Scalar> operator*(RxSO3<Scalar> const& other) {
      return RxSO3<Scalar>(unit_quaternion * other.unit_quaternion, scale * other.scale);
    }

    EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
      const Quaternion& q = unit_quaternion;
      Point uv = q.vec().cross(p); uv += uv;
      return scale * (p + q.w()*uv + q.vec().cross(uv));
    }

    EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
      Point4 p1; p1 << this->operator*(p.template segment<3>(0)), p(3);
      return p1;
    }

    EIGEN_DEVICE_FUNC Adjoint Adj() const {
      Adjoint Ad = Adjoint::Identity();
      Ad.template block<3,3>(0,0) = unit_quaternion.toRotationMatrix();
      return Ad;
    }

    EIGEN_DEVICE_FUNC Transformation Matrix() const {
      return scale * unit_quaternion.toRotationMatrix();
    }

    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,4,4> Matrix4x4() const {
      Eigen::Matrix<Scalar,4,4> T;
      T = Eigen::Matrix<Scalar,4,4>::Identity();
      T.template block<3,3>(0,0) = Matrix();
      return T;
    }

    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,5,5> orthogonal_projector() const {
      // jacobian action on a point
      Eigen::Matrix<Scalar,5,5> J = Eigen::Matrix<Scalar,5,5>::Zero();
      
      J.template block<3,3>(0,0) = 0.5 * (
        unit_quaternion.w() * Matrix3::Identity() + 
        SO3<Scalar>::hat(-unit_quaternion.vec())
      );
      
      J.template block<1,3>(3,0) = 0.5 * (-unit_quaternion.vec());

      // scale
      J(4,3) = scale;

      return J;
    }

    EIGEN_DEVICE_FUNC Transformation Rotation() const {
      return unit_quaternion.toRotationMatrix();
    }

    EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const {
      return Adj() * a;
    }

    EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const {
      return Adj().transpose() * a;
    }

    EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& phi_sigma) {
      Vector3 const phi = phi_sigma.template segment<3>(0);
      return SO3<Scalar>::hat(phi) + phi(3) * Transformation::Identity();
    }

    EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& phi_sigma) {
      Vector3 const phi = phi_sigma.template segment<3>(0);
      Matrix3 const Phi = SO3<Scalar>::hat(phi);

      Adjoint ad = Adjoint::Zero();
      ad.template block<3,3>(0,0) = Phi;

      return ad;
    }

    EIGEN_DEVICE_FUNC Tangent Log() const {
      using std::abs;
      using std::atan;
      using std::sqrt;

      Scalar squared_n = unit_quaternion.vec().squaredNorm();
      Scalar w = unit_quaternion.w();
      Scalar two_atan_nbyw_by_n;

      /// Atan-based log thanks to
      ///
      /// C. Hertzberg et al.:
      /// "Integrating Generic Sensor Fusion Algorithms with Sound State
      /// Representation through Encapsulation of Manifolds"
      /// Information Fusion, 2011

      if (squared_n < EPS * EPS) {
        two_atan_nbyw_by_n = Scalar(2) / w - Scalar(2.0/3.0) * (squared_n) / (w * w * w);
      } else {
        Scalar n = sqrt(squared_n);
        if (abs(w) < EPS) {
          if (w > Scalar(0)) {
            two_atan_nbyw_by_n = PI / n;
          } else {
            two_atan_nbyw_by_n = -PI / n;
          }
        } else {
          two_atan_nbyw_by_n = Scalar(2) * atan(n / w) / n;
        }
      }

      Tangent phi_sigma;
      phi_sigma << two_atan_nbyw_by_n * unit_quaternion.vec(), log(scale);

      return phi_sigma;
    }

    EIGEN_DEVICE_FUNC static RxSO3<Scalar> Exp(Tangent const& phi_sigma) {
      Vector3 phi = phi_sigma.template segment<3>(0);
      Scalar scale = exp(phi_sigma(3));

      Scalar theta2 = phi.squaredNorm();
      Scalar theta = sqrt(theta2);
      Scalar imag_factor;
      Scalar real_factor;

      if (theta < EPS) {
        Scalar theta4 = theta2 * theta2;
        imag_factor = Scalar(0.5) - Scalar(1.0/48.0) * theta2 + Scalar(1.0/3840.0) * theta4;
        real_factor = Scalar(1) - Scalar(1.0/8.0) * theta2 + Scalar(1.0/384.0) * theta4;
      } else {
        imag_factor = sin(.5 * theta) / theta;
        real_factor = cos(.5 * theta);
      }

      Quaternion q(real_factor, imag_factor*phi.x(), imag_factor*phi.y(), imag_factor*phi.z());
      return RxSO3<Scalar>(q, scale);
    }
    
    EIGEN_DEVICE_FUNC static Matrix3 calcW(Tangent const& phi_sigma) {
      // left jacobian
      using std::abs;
      Matrix3 const I = Matrix3::Identity();
      Scalar const one(1);
      Scalar const half(0.5);
      
      Vector3 const phi = phi_sigma.template segment<3>(0);
      Scalar const sigma = phi_sigma(3);
      Scalar const theta = phi.norm();

      Matrix3 const Phi = SO3<Scalar>::hat(phi);
      Matrix3 const Phi2 = Phi * Phi;
      Scalar const scale = exp(sigma);

      Scalar A, B, C;
      if (abs(sigma) < EPS) {
        C = one;
        if (abs(theta) < EPS) {
          A = half;
          B = Scalar(1. / 6.);
        } else {
          Scalar theta_sq = theta * theta;
          A = (one - cos(theta)) / theta_sq;
          B = (theta - sin(theta)) / (theta_sq * theta);
        }
      } else {
        C = (scale - one) / sigma;
        if (abs(theta) < EPS) {
          Scalar sigma_sq = sigma * sigma;
          A = ((sigma - one) * scale + one) / sigma_sq;
          B = (scale * half * sigma_sq + scale - one - sigma * scale) /
              (sigma_sq * sigma);
        } else {
          Scalar theta_sq = theta * theta;
          Scalar a = scale * sin(theta);
          Scalar b = scale * cos(theta);
          Scalar c = theta_sq + sigma * sigma;
          A = (a * sigma + (one - b) * theta) / (theta * c);
          B = (C - ((b - one) * sigma + a * theta) / (c)) * one / (theta_sq);
        }
      }
      return A * Phi + B * Phi2 + C * I;
    }

    EIGEN_DEVICE_FUNC static Matrix3 calcWInv(Tangent const& phi_sigma) {
      // left jacobian inverse
      Matrix3 const I = Matrix3::Identity();
      Scalar const half(0.5);
      Scalar const one(1);
      Scalar const two(2);
      
      Vector3 const phi = phi_sigma.template segment<3>(0);
      Scalar const sigma = phi_sigma(3);
      Scalar const theta = phi.norm();
      Scalar const scale = exp(sigma);

      Matrix3 const Phi = SO3<Scalar>::hat(phi);
      Matrix3 const Phi2 = Phi * Phi;
      Scalar const scale_sq = scale * scale;
      Scalar const theta_sq = theta * theta;
      Scalar const sin_theta = sin(theta);
      Scalar const cos_theta = cos(theta);

      Scalar a, b, c;
      if (abs(sigma * sigma) < EPS) {
        c = one - half * sigma;
        a = -half;
        if (abs(theta_sq) < EPS) {
          b = Scalar(1. / 12.);
        } else {
          b = (theta * sin_theta + two * cos_theta - two) /
              (two * theta_sq * (cos_theta - one));
        }
      } else {
        Scalar const scale_cu = scale_sq * scale;
        c = sigma / (scale - one);
        if (abs(theta_sq) < EPS) {
          a = (-sigma * scale + scale - one) / ((scale - one) * (scale - one));
          b = (scale_sq * sigma - two * scale_sq + scale * sigma + two * scale) /
              (two * scale_cu - Scalar(6) * scale_sq + Scalar(6) * scale - two);
        } else {
          Scalar const s_sin_theta = scale * sin_theta;
          Scalar const s_cos_theta = scale * cos_theta;
          a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
              (theta * (scale_sq - two * s_cos_theta + one));
          b = -scale *
              (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
              scale * sigma + sigma * cos_theta - sigma) /
              (theta_sq * (scale_cu - two * scale * s_cos_theta - scale_sq +
                          two * s_cos_theta + scale - one));
        }
      }
      return a * Phi + b * Phi2 + c * I;
    }

    EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& phi_sigma) {
      // left jacobian
      Adjoint J = Adjoint::Identity();
      Vector3 phi = phi_sigma.template segment<3>(0);
      J.template block<3,3>(0,0) = SO3<Scalar>::left_jacobian(phi);
      return J;
    }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& phi_sigma) {
      // left jacobian inverse
      Adjoint Jinv = Adjoint::Identity();
      Vector3 phi = phi_sigma.template segment<3>(0);
      Jinv.template block<3,3>(0,0) = SO3<Scalar>::left_jacobian_inverse(phi);
      return Jinv;
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,3,4> act_jacobian(Point const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,3,4> Ja; 
      Ja << SO3<Scalar>::hat(-p), p;
      return Ja;
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,4,4> act4_jacobian(Point4 const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,4,4> J = Eigen::Matrix<Scalar,4,4>::Zero();
      J.template block<3,3>(0,0) = SO3<Scalar>::hat(-p.template segment<3>(0));
      J.template block<3,1>(0,3) = p.template segment<3>(0); 
      return J;
    }

  private:
    Quaternion unit_quaternion;
    Scalar scale;
};

#endif


