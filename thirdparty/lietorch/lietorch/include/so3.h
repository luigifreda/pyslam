
#ifndef SO3_HEADER
#define SO3_HEADER

#include <cuda.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include "common.h"

template <typename Scalar>
class SO3 {
  public:
    const static int constexpr K = 3; // manifold dimension
    const static int constexpr N = 4; // embedding dimension

    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector4 = Eigen::Matrix<Scalar,4,1>;
    using Matrix3 = Eigen::Matrix<Scalar,3,3>;

    using Tangent = Eigen::Matrix<Scalar,K,1>;
    using Data = Eigen::Matrix<Scalar,N,1>;

    using Point = Eigen::Matrix<Scalar,3,1>;
    using Point4 = Eigen::Matrix<Scalar,4,1>;
    using Transformation = Eigen::Matrix<Scalar,3,3>;
    using Adjoint = Eigen::Matrix<Scalar,K,K>;
    using Quaternion = Eigen::Quaternion<Scalar>;

    EIGEN_DEVICE_FUNC SO3(Quaternion const& q) : unit_quaternion(q) {
      unit_quaternion.normalize();
    };

    EIGEN_DEVICE_FUNC SO3(const Scalar *data) : unit_quaternion(data) {
      unit_quaternion.normalize();
    };

    EIGEN_DEVICE_FUNC SO3() {
      unit_quaternion = Quaternion::Identity();
    }

    EIGEN_DEVICE_FUNC SO3<Scalar> inv() {
      return SO3<Scalar>(unit_quaternion.conjugate());
    }

    EIGEN_DEVICE_FUNC Data data() const {
      return unit_quaternion.coeffs();
    }

    EIGEN_DEVICE_FUNC SO3<Scalar> operator*(SO3<Scalar> const& other) {
      return SO3(unit_quaternion * other.unit_quaternion);
    }

    EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
      const Quaternion& q = unit_quaternion;
      Point uv = q.vec().cross(p);
      uv += uv;
      return p + q.w()*uv + q.vec().cross(uv);
    }

    EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
      Point4 p1; p1 << this->operator*(p.template segment<3>(0)), p(3);
      return p1;
    }
    
    EIGEN_DEVICE_FUNC Adjoint Adj() const {
      return unit_quaternion.toRotationMatrix();
    }

    EIGEN_DEVICE_FUNC Transformation Matrix() const {
      return unit_quaternion.toRotationMatrix();
    }

    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,4,4> Matrix4x4() const {
      Eigen::Matrix<Scalar,4,4> T = Eigen::Matrix<Scalar,4,4>::Identity();
      T.template block<3,3>(0,0) = Matrix();
      return T;
    }

    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,4,4> orthogonal_projector() const {
      // jacobian action on a point
      Eigen::Matrix<Scalar,4,4> J = Eigen::Matrix<Scalar,4,4>::Zero();
      J.template block<3,3>(0,0) = 0.5 * (
        unit_quaternion.w() * Matrix3::Identity() + 
        SO3<Scalar>::hat(-unit_quaternion.vec())
      );
      
      J.template block<1,3>(3,0) = 0.5 * (-unit_quaternion.vec());
      return J;
    }

    EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const {
      return Adj() * a;
    }

    EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const {
      return Adj().transpose() * a;
    }

    EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& phi) {
      Transformation Phi;
      Phi << 
        0.0, -phi(2), phi(1), 
        phi(2), 0.0, -phi(0), 
        -phi(1), phi(0), 0.0;

      return Phi;
    }

    EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& phi) {
      return SO3<Scalar>::hat(phi);
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
        // If quaternion is normalized and n=0, then w should be 1;
        // w=0 should never happen here!
        Scalar squared_w = w * w;
        two_atan_nbyw_by_n =
            Scalar(2) / w - Scalar(2.0/3.0) * (squared_n) / (w * squared_w);
      } else {
        Scalar n = sqrt(squared_n);
        if (abs(w) < EPS) {
          if (w > Scalar(0)) {
            two_atan_nbyw_by_n = Scalar(PI) / n;
          } else {
            two_atan_nbyw_by_n = -Scalar(PI) / n;
          }
        } else {
          two_atan_nbyw_by_n = Scalar(2) * atan(n / w) / n;
        }
      }

      return two_atan_nbyw_by_n * unit_quaternion.vec();
    }

    EIGEN_DEVICE_FUNC static SO3<Scalar> Exp(Tangent const& phi) {
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
      return SO3<Scalar>(q);
    }
    
    EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& phi) {
      // left jacobian
      Matrix3 I = Matrix3::Identity();
      Matrix3 Phi = SO3<Scalar>::hat(phi);
      Matrix3 Phi2 = Phi * Phi;

      Scalar theta2 = phi.squaredNorm();
      Scalar theta = sqrt(theta2);

      Scalar coef1 = (theta < EPS) ? 
        Scalar(1.0/2.0) - Scalar(1.0/24.0) * theta2 : 
        (1.0 - cos(theta)) / theta2;
      
      Scalar coef2 = (theta < EPS) ? 
        Scalar(1.0/6.0) - Scalar(1.0/120.0) * theta2 : 
        (theta - sin(theta)) / (theta2 * theta); 

      return I + coef1 * Phi + coef2 * Phi2;
    }

    EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& phi) {
      // left jacobian inverse
      Matrix3 I = Matrix3::Identity();
      Matrix3 Phi = SO3<Scalar>::hat(phi);
      Matrix3 Phi2 = Phi * Phi;

      Scalar theta2 = phi.squaredNorm();
      Scalar theta = sqrt(theta2);
      Scalar half_theta = Scalar(.5) * theta ;

      Scalar coef2 = (theta < EPS) ? Scalar(1.0/12.0) : 
           (Scalar(1) -
            theta * cos(half_theta) / (Scalar(2) * sin(half_theta))) /
               (theta * theta);

      return I + Scalar(-0.5) * Phi + coef2 * Phi2;
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,3,3> act_jacobian(Point const& p) {
      // jacobian action on a point
      return SO3<Scalar>::hat(-p);
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,4,3> act4_jacobian(Point4 const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,4,3> J = Eigen::Matrix<Scalar,4,3>::Zero();
      J.template block<3,3>(0,0) = SO3<Scalar>::hat(-p.template segment<3>(0));
      return J;
    }

  private:
    Quaternion unit_quaternion;

};

#endif


