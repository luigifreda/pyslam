/**
 * This file is part of S-PTAM.
 *
 * Copyright (C) 2013-2017 Taihú Pire
 * Copyright (C) 2014-2017 Thomas Fischer
 * Copyright (C) 2016-2017 Gastón Castro
 * Copyright (C) 2017 Matias Nitsche
 * For more information see <https://github.com/lrse/sptam>
 *
 * S-PTAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * S-PTAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with S-PTAM. If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:  Taihú Pire
 *           Thomas Fischer
 *           Gastón Castro
 *           Matías Nitsche
 *
 * Laboratory of Robotics and Embedded Systems
 * Department of Computer Science
 * Faculty of Exact and Natural Sciences
 * University of Buenos Aires
 */

#include "types_stereo_sba.hpp"

namespace g2o {

using namespace std;

// point to camera projection, monocular
EdgeProjectP2MCRight::EdgeProjectP2MCRight() :
BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexCam>()
{
  information().setIdentity();
}

bool EdgeProjectP2MCRight::read(std::istream& is)
{
  // measured keypoint
  for (int i=0; i<2; i++)
    is >> _measurement[i];
  setMeasurement(_measurement);
  // information matrix is the identity for features, could be changed to allow arbitrary covariances
  information().setIdentity();
  return true;
}

bool EdgeProjectP2MCRight::write(std::ostream& os) const
{
  for (int i=0; i<2; i++)
    os  << measurement()[i] << " ";
  return os.good();
}

/**
 * \brief Jacobian for monocular projection
 */
  void EdgeProjectP2MCRight::linearizeOplus()
  {
    VertexCam *vc = static_cast<VertexCam *>(_vertices[1]);
    const SBACam &cam = vc->estimate();

    Vector3d pb(cam.baseline,0,0);

    VertexSBAPointXYZ *vp = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
    Vector4d pt, trans;
    pt.head<3>() = vp->estimate();
    pt(3) = 1.0;
    Quaterniond rotation = cam.rotation(); // camera orientation
//    trans.head<3>() = rotation.inverse() * pb - rotation.inverse() * cam.translation(); // right camera position
    trans.head<3>() = rotation * pb + cam.translation(); // right camera position

    trans(3) = 1.0;

    // first get the world point in right camera coords
    Eigen::Matrix<double,3,1> pc = (cam.w2n * pt) - pb;

    // Jacobiano con respecto a los parámetros de la camara

    // Jacobians wrt camera parameters
    // set d(quat-x) values [ pz*dpx/dx - px*dpz/dx ] / pz^2
    double px = pc(0);
    double py = pc(1);
    double pz = pc(2);
    double ipz2 = 1.0/(pz*pz);
    if (g2o_isnan(ipz2) ) {
      std::cout << "[SetJac] infinite jac" << std::endl;
      abort();
    }

    double ipz2fx = ipz2*cam.Kcam(0,0); // Fx
    double ipz2fy = ipz2*cam.Kcam(1,1); // Fy

    Eigen::Matrix<double,3,1> pwt;

    // check for local vars
    pwt = (pt-trans).head<3>(); // transform translations, use differential rotation

    // utiliza rotaciones diferenciales
    // en la funcion sbacam.setDr() se calculan las rotaciones diferenciales

    // dx
    Eigen::Matrix<double,3,1> dp = cam.dRdx * pwt; // dR'/dq * [pw - t]
    _jacobianOplusXj(0,3) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXj(1,3) = (pz*dp(1) - py*dp(2))*ipz2fy;
    // dy
    dp = cam.dRdy * pwt; // dR'/dq * [pw - t]
    _jacobianOplusXj(0,4) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXj(1,4) = (pz*dp(1) - py*dp(2))*ipz2fy;
    // dz
    dp = cam.dRdz * pwt; // dR'/dq * [pw - t]
    _jacobianOplusXj(0,5) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXj(1,5) = (pz*dp(1) - py*dp(2))*ipz2fy;


    // set d(t) values [ pz*dpx/dx - px*dpz/dx ] / pz^2
    dp = -cam.w2n.col(0);        // dpc / dx
    _jacobianOplusXj(0,0) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXj(1,0) = (pz*dp(1) - py*dp(2))*ipz2fy;
    dp = -cam.w2n.col(1);        // dpc / dy
    _jacobianOplusXj(0,1) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXj(1,1) = (pz*dp(1) - py*dp(2))*ipz2fy;
    dp = -cam.w2n.col(2);        // dpc / dz
    _jacobianOplusXj(0,2) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXj(1,2) = (pz*dp(1) - py*dp(2))*ipz2fy;

    // aca esta calculando el jacobiano con respecto al punto en el mundo

    // Jacobians wrt point parameters
    // set d(t) values [ pz*dpx/dx - px*dpz/dx ] / pz^2
    dp = cam.w2n.col(0); // dpc / dx
    _jacobianOplusXi(0,0) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXi(1,0) = (pz*dp(1) - py*dp(2))*ipz2fy;
    dp = cam.w2n.col(1); // dpc / dy
    _jacobianOplusXi(0,1) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXi(1,1) = (pz*dp(1) - py*dp(2))*ipz2fy;
    dp = cam.w2n.col(2); // dpc / dz
    _jacobianOplusXi(0,2) = (pz*dp(0) - px*dp(2))*ipz2fx;
    _jacobianOplusXi(1,2) = (pz*dp(1) - py*dp(2))*ipz2fy;
  }

} // g2o
