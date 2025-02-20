
#ifndef Sim3_HEADER
#define Sim3_HEADER

#include <stdio.h>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include "common.h"
#include "so3.h"
#include "rxso3.h"


template <typename Scalar>
class Sim3 {
  public:
    const static int constexpr K = 7; // manifold dimension
    const static int constexpr N = 8; // embedding dimension

    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector4 = Eigen::Matrix<Scalar,4,1>;
    using Matrix3 = Eigen::Matrix<Scalar,3,3>;

    using Tangent = Eigen::Matrix<Scalar,K,1>;
    using Point = Eigen::Matrix<Scalar,3,1>;
    using Point4 = Eigen::Matrix<Scalar,4,1>;
    using Data = Eigen::Matrix<Scalar,N,1>;
    using Transformation = Eigen::Matrix<Scalar,4,4>;
    using Adjoint = Eigen::Matrix<Scalar,K,K>;

    EIGEN_DEVICE_FUNC Sim3() {
      translation = Vector3::Zero();
    }

    EIGEN_DEVICE_FUNC Sim3(RxSO3<Scalar> const& rxso3, Vector3 const& t)
      : rxso3(rxso3), translation(t) {};

    EIGEN_DEVICE_FUNC Sim3(const Scalar *data) 
      : translation(data), rxso3(data+3)  {};

    EIGEN_DEVICE_FUNC Sim3<Scalar> inv() {
      return Sim3<Scalar>(rxso3.inv(), -(rxso3.inv() * translation));
    }

    EIGEN_DEVICE_FUNC Data data() const {
      Data data_vec; data_vec << translation, rxso3.data();
      return data_vec;
    }

    EIGEN_DEVICE_FUNC Sim3<Scalar> operator*(Sim3<Scalar> const& other) {
      return Sim3(rxso3 * other.rxso3, translation + rxso3 * other.translation);
    }

    EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
      return (rxso3 * p) + translation;
    }
    
    EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
      Point4 p1; p1 << rxso3 * p.template segment<3>(0) + p(3) * translation , p(3);
      return p1;
    }

    EIGEN_DEVICE_FUNC Transformation Matrix() const {
      Transformation T = Transformation::Identity();
      T.template block<3,3>(0,0) = rxso3.Matrix();
      T.template block<3,1>(0,3) = translation;
      return T;
    }

    EIGEN_DEVICE_FUNC Transformation Matrix4x4() const {
      Transformation T = Transformation::Identity();
      T.template block<3,3>(0,0) = rxso3.Matrix();
      T.template block<3,1>(0,3) = translation;
      return T;
    }

    EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar,8,8> orthogonal_projector() const {
      // jacobian action on a point
      Eigen::Matrix<Scalar,8,8> J = Eigen::Matrix<Scalar,8,8>::Zero();
      J.template block<3,3>(0,0) = Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-translation);
      J.template block<3,1>(0,6) = translation;
      J.template block<5,5>(3,3) = rxso3.orthogonal_projector();
      return J;
    }

    EIGEN_DEVICE_FUNC Adjoint Adj() const {
      Adjoint Ad = Adjoint::Identity();
      Matrix3 sR = rxso3.Matrix();
      Matrix3 tx = SO3<Scalar>::hat(translation);
      Matrix3 R = rxso3.Rotation();

      Ad.template block<3,3>(0,0) = sR;
      Ad.template block<3,3>(0,3) = tx * R;
      Ad.template block<3,1>(0,6) = -translation;
      Ad.template block<3,3>(3,3) = R;

      return Ad;
    }

    EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const {
      return Adj() * a;
    }

    EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const {
      return Adj().transpose() * a;
    }

    EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& tau_phi_sigma) {
      Vector3 tau = tau_phi_sigma.template segment<3>(0);
      Vector3 phi = tau_phi_sigma.template segment<3>(3);
      Scalar sigma = tau_phi_sigma(6);

      Matrix3 Phi = SO3<Scalar>::hat(phi);
      Matrix3 I = Matrix3::Identity();
      
      Transformation Omega = Transformation::Zero();
      Omega.template block<3,3>(0,0) = Phi + sigma * I;
      Omega.template block<3,1>(0,3) = tau;
      
      return Omega;
    }

    EIGEN_DEVICE_FUNC  static Adjoint adj(Tangent const& tau_phi_sigma) {
      Adjoint ad = Adjoint::Zero();
      Vector3 tau = tau_phi_sigma.template segment<3>(0);
      Vector3 phi = tau_phi_sigma.template segment<3>(3);
      Scalar sigma = tau_phi_sigma(6);

      Matrix3 Tau = SO3<Scalar>::hat(tau);
      Matrix3 Phi = SO3<Scalar>::hat(phi);
      Matrix3 I = Matrix3::Identity();

      ad.template block<3,3>(0,0) = Phi + sigma * I;
      ad.template block<3,3>(0,3) = Tau;
      ad.template block<3,1>(0,6) = -tau;
      ad.template block<3,3>(3,3) = Phi;

      return ad;
    }
    

    EIGEN_DEVICE_FUNC Tangent Log() const {
      // logarithm map
      Vector4 phi_sigma = rxso3.Log();      
      Matrix3 W = RxSO3<Scalar>::calcW(phi_sigma);
      
      Tangent tau_phi_sigma; 
      tau_phi_sigma << W.inverse() * translation, phi_sigma;

      return tau_phi_sigma;
    }

    EIGEN_DEVICE_FUNC static Sim3<Scalar> Exp(Tangent const& tau_phi_sigma) {
      // exponential map
      Vector3 tau = tau_phi_sigma.template segment<3>(0);
      Vector4 phi_sigma = tau_phi_sigma.template segment<4>(3);
      
      RxSO3<Scalar> rxso3 = RxSO3<Scalar>::Exp(phi_sigma);
      Matrix3 W = RxSO3<Scalar>::calcW(phi_sigma);

      return Sim3<Scalar>(rxso3, W*tau);
    }

    EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& tau_phi_sigma) {
      // left jacobian
      Adjoint const Xi = adj(tau_phi_sigma);
      Adjoint const Xi2 = Xi * Xi;
      Adjoint const Xi4 = Xi2 * Xi2;

      return Adjoint::Identity() 
        + Scalar(1.0/2.0)*Xi
        + Scalar(1.0/6.0)*Xi2
        + Scalar(1.0/24.0)*Xi*Xi2
        + Scalar(1.0/120.0)*Xi4;
        + Scalar(1.0/720.0)*Xi*Xi4;
    }

    EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& tau_phi_sigma) {
      // left jacobian inverse
      Adjoint const Xi = adj(tau_phi_sigma);
      Adjoint const Xi2 = Xi * Xi;
      Adjoint const Xi4 = Xi2 * Xi2;

      return Adjoint::Identity() 
        - Scalar(1.0/2.0)*Xi
        + Scalar(1.0/12.0)*Xi2
        - Scalar(1.0/720.0)*Xi4;
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,3,7> act_jacobian(Point const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,3,7> J;
      J.template block<3,3>(0,0) = Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-p);
      J.template block<3,1>(0,6) = p;
      return J;
    }

    EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar,4,7> act4_jacobian(Point4 const& p) {
      // jacobian action on a point
      Eigen::Matrix<Scalar,4,7> J = Eigen::Matrix<Scalar,4,7>::Zero();
      J.template block<3,3>(0,0) = p(3) * Matrix3::Identity();
      J.template block<3,3>(0,3) = SO3<Scalar>::hat(-p.template segment<3>(0));
      J.template block<3,1>(0,6) = p.template segment<3>(0);
      return J;
    }

  private:
    Vector3 translation;
    RxSO3<Scalar> rxso3;
};

#endif

