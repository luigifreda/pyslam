
#ifndef SE3_HEADER
#define SE3_HEADER

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include "common.h"
#include "so3.h"


template <typename Scalar>
class SE3 {
  public:
    const static int constexpr K = 6; // manifold dimension
    const static int constexpr N = 7; // embedding dimension

    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector4 = Eigen::Matrix<Scalar,4,1>;
    using Matrix3 = Eigen::Matrix<Scalar,3,3>;

    using Tangent = Eigen::Matrix<Scalar,K,1>;
    using Point = Eigen::Matrix<Scalar,3,1>;
    using Point4 = Eigen::Matrix<Scalar,4,1>;
    using Data = Eigen::Matrix<Scalar,N,1>;
    using Transformation = Eigen::Matrix<Scalar,4,4>;
    using Adjoint = Eigen::Matrix<Scalar,K,K>;

    EIGEN_DEVICE_FUNC SE3() { translation = Vector3::Zero(); }

    EIGEN_DEVICE_FUNC SE3(SO3<Scalar> const& so3, Vector3 const& t) : so3(so3), translation(t) {};

    EIGEN_DEVICE_FUNC SE3(const Scalar *data) :  translation(data), so3(data+3) {};

    EIGEN_DEVICE_FUNC SE3<Scalar> inv() {
      return SE3(so3.inv(), -(so3.inv()*translation));
    }

    EIGEN_DEVICE_FUNC Data data() const {
      Data data_vec; data_vec << translation, so3.data();
      return data_vec;
    }

    EIGEN_DEVICE_FUNC SE3<Scalar> operator*(SE3<Scalar> const& other) {
      return SE3(so3 * other.so3, translation + so3 * other.translation);
    }

    EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
      return so3 * p + translation;
    }

    EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
      Point4 p1; p1 << so3 * p.template segment<3>(0) + translation * p(3), p(3);
      return p1;
    }

    EIGEN_DEVICE_FUNC Adjoint Adj() const {
      Matrix3 R = so3.Matrix();
      Matrix3 tx = SO3<Scalar>::hat(translation);
      Matrix3 Zer = Matrix3::Zero();

      Adjoint Ad;
      Ad << R, tx*R, Zer, R;

      return Ad;
    }

    EIGEN_DEVICE_FUNC Transformation Matrix() const {
      Transformation T = Transformation::Identity();
      T.template block<3,3>(0,0) = so3.Matrix();
      T.template block<3,1>(0,3) = translation;
      return T;
    }

    EIGEN_DEVICE_FUNC Transformation Matrix4x4() const {
      return Matrix();
    }

    EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const {
      return Adj() * a;
    }

    EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const {
      return Adj().transpose() * a;
    }


    EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& tau_phi) {
      Vector3 tau = tau_phi.template segment<3>(0);
      Vector3 phi = tau_phi.template segment<3>(3);

      Transformation TauPhi = Transformation::Zero();
      TauPhi.template block<3,3>(0,0) = SO3<Scalar>::hat(phi);
      TauPhi.template block<3,1>(0,3) = tau;
      
      return TauPhi;
    }

    EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& tau_phi) {
      Vector3 tau = tau_phi.template segment<3>(0);
      Vector3 phi = tau_phi.template segment<3>(3);

      Matrix3 Tau = SO3<Scalar>::hat(tau);
      Matrix3 Phi = SO3<Scalar>::hat(phi);
      Matrix3 Zer = Matrix3::Zero();

      Adjoint ad;
      ad << Phi, Tau, Zer, Phi;

      return ad;
    }

    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,7,7> orthogonal_projector() const {
      // jacobian action on a point
      Eigen::Matrix<Scalar,7,7> J = Eigen::Matrix<Scalar,7,7>::Zero();
      J.template block<3,3>(0,0) = Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-translation);
      J.template block<4,4>(3,3) = so3.orthogonal_projector();

      return J;
    }

    EIGEN_DEVICE_FUNC Tangent Log() const {
      Vector3 phi = so3.Log();      
      Matrix3 Vinv = SO3<Scalar>::left_jacobian_inverse(phi);

      Tangent tau_phi; 
      tau_phi << Vinv * translation, phi;

      return tau_phi;
    }

    EIGEN_DEVICE_FUNC static SE3<Scalar> Exp(Tangent const& tau_phi) {
      Vector3 tau = tau_phi.template segment<3>(0);
      Vector3 phi = tau_phi.template segment<3>(3);

      SO3<Scalar> so3 = SO3<Scalar>::Exp(phi);
      Vector3 t = SO3<Scalar>::left_jacobian(phi) * tau;

      return SE3<Scalar>(so3, t);
    }

    EIGEN_DEVICE_FUNC static Matrix3 calcQ(Tangent const& tau_phi) {
      // Q matrix
      Vector3 tau = tau_phi.template segment<3>(0);
      Vector3 phi = tau_phi.template segment<3>(3);
      Matrix3 Tau = SO3<Scalar>::hat(tau);
      Matrix3 Phi = SO3<Scalar>::hat(phi);

      Scalar theta = phi.norm();
      Scalar theta_pow2 = theta * theta;
      Scalar theta_pow4 = theta_pow2 * theta_pow2;

      Scalar coef1 = (theta < EPS) ?
        Scalar(1.0/6.0) - Scalar(1.0/120.0) * theta_pow2 : 
        (theta - sin(theta)) / (theta_pow2 * theta);

      Scalar coef2 = (theta < EPS) ?
        Scalar(1.0/24.0) - Scalar(1.0/720.0) * theta_pow2 : 
        (theta_pow2 + 2*cos(theta) - 2) / (2 * theta_pow4);

      Scalar coef3 = (theta < EPS) ?
        Scalar(1.0/120.0) - Scalar(1.0/2520.0) * theta_pow2 : 
        (2*theta - 3*sin(theta) + theta*cos(theta)) / (2 * theta_pow4 * theta);

      Matrix3 Q = Scalar(0.5) * Tau + 
        coef1 * (Phi*Tau + Tau*Phi + Phi*Tau*Phi) +
        coef2 * (Phi*Phi*Tau + Tau*Phi*Phi - 3*Phi*Tau*Phi) + 
        coef3 * (Phi*Tau*Phi*Phi + Phi*Phi*Tau*Phi);

      return Q;
    }
    
    EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& tau_phi) {
      // left jacobian
      Vector3 phi = tau_phi.template segment<3>(3);
      Matrix3 J = SO3<Scalar>::left_jacobian(phi);
      Matrix3 Q = SE3<Scalar>::calcQ(tau_phi);
      Matrix3 Zer = Matrix3::Zero();

      Adjoint J6x6;
      J6x6 << J, Q, Zer, J;

      return J6x6;
    }

    EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& tau_phi) {
      // left jacobian inverse
      Vector3 tau = tau_phi.template segment<3>(0);
      Vector3 phi = tau_phi.template segment<3>(3);
      Matrix3 Jinv = SO3<Scalar>::left_jacobian_inverse(phi);
      Matrix3 Q = SE3<Scalar>::calcQ(tau_phi);
      Matrix3 Zer = Matrix3::Zero();

      Adjoint J6x6;
      J6x6 << Jinv, -Jinv * Q * Jinv, Zer, Jinv;

      return J6x6;

    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,3,6> act_jacobian(Point const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,3,6> J;
      J.template block<3,3>(0,0) = Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-p);
      return J;
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,4,6> act4_jacobian(Point4 const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,4,6> J = Eigen::Matrix<Scalar,4,6>::Zero();
      J.template block<3,3>(0,0) = p(3) * Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-p.template segment<3>(0));
      return J;
    }




  private:
    SO3<Scalar> so3;
    Vector3 translation;

};

#endif

