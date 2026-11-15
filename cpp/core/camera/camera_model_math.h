/*
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

// Independent camera projection math (pinhole, Kannala-Brandt4, rational-radial
// pinhole, double sphere). This is the only copy of the formulas.
// cpp_core and the Python-facing cpp/modules/camera_models module both include it.
// Jacobians are d(projection)/d(Xc). Pose and point Jacobians are composed
// afterwards so the same 2x3 block serves g2o (left perturbation) and gtsam
// (whatever dXc/dpose the factor supplies).

#include <Eigen/Dense>
#include <algorithm>

#include <cmath>
#include <limits>

namespace pyslam {
namespace camera_model {

template <typename Scalar> using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
template <typename Scalar> using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar> using Mat23 = Eigen::Matrix<Scalar, 2, 3>;
template <typename Scalar> using Mat32 = Eigen::Matrix<Scalar, 3, 2>;
template <typename Scalar> using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
template <typename Scalar> using Mat33 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar> using Mat26 = Eigen::Matrix<Scalar, 2, 6>;
template <typename Scalar> using Mat36 = Eigen::Matrix<Scalar, 3, 6>;
template <typename Scalar> using Mat63 = Eigen::Matrix<Scalar, 6, 3>;

template <typename Scalar> inline Scalar domain_epsilon() {
    return std::sqrt(std::numeric_limits<Scalar>::epsilon());
}

// ---------------------------------------------------------------------------
// Pinhole, no distortion. Unproject stays on the plane z = 1.
// ---------------------------------------------------------------------------

template <typename Scalar>
inline bool project_pinhole(Scalar fx, Scalar fy, Scalar cx, Scalar cy, const Vec3<Scalar> &Xc,
                            Vec2<Scalar> &uv, Mat23<Scalar> *J_uv_Xc = nullptr) {
    const Scalar x = Xc[0];
    const Scalar y = Xc[1];
    const Scalar z = Xc[2];
    const bool valid = Xc.allFinite() && z > domain_epsilon<Scalar>();
    if (!valid) {
        uv << cx, cy;
        if (J_uv_Xc) J_uv_Xc->setZero();
        return false;
    }
    const Scalar inv_z = Scalar(1) / z;
    uv[0] = fx * x * inv_z + cx;
    uv[1] = fy * y * inv_z + cy;
    if (J_uv_Xc) {
        const Scalar inv_z2 = inv_z * inv_z;
        J_uv_Xc->setZero();
        (*J_uv_Xc)(0, 0) = fx * inv_z;
        (*J_uv_Xc)(0, 2) = -fx * x * inv_z2;
        (*J_uv_Xc)(1, 1) = fy * inv_z;
        (*J_uv_Xc)(1, 2) = -fy * y * inv_z2;
    }
    return valid;
}

template <typename Scalar>
inline bool unproject_pinhole_bearing(Scalar fx, Scalar fy, Scalar cx, Scalar cy,
                                      const Vec2<Scalar> &uv, Vec3<Scalar> &bearing,
                                      Mat32<Scalar> *J_bearing_uv = nullptr) {
    const Scalar mx = (uv[0] - cx) / fx;
    const Scalar my = (uv[1] - cy) / fy;
    bearing << mx, my, Scalar(1);
    if (J_bearing_uv) {
        J_bearing_uv->setZero();
        (*J_bearing_uv)(0, 0) = Scalar(1) / fx;
        (*J_bearing_uv)(1, 1) = Scalar(1) / fy;
    }
    return true;
}

// d(u, v, u - bf/z) / dXc from the mono pinhole Jacobian.
template <typename Scalar>
inline void pinhole_stereo_jacobian(const Mat23<Scalar> &J_uv, const Vec3<Scalar> &Xc, Scalar bf,
                                    Eigen::Matrix<Scalar, 3, 3> &J_uvu) {
    J_uvu.template topRows<2>() = J_uv;
    J_uvu.row(2) = J_uv.row(0);
    const Scalar z = Xc[2];
    J_uvu(2, 2) += bf / (z * z);
}

// ---------------------------------------------------------------------------
// Kannala-Brandt with four distortion coefficients:
//   d(theta) = theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9
//   theta = atan2(r, z), r = hypot(x, y)
// ---------------------------------------------------------------------------

template <typename Scalar>
inline Scalar kb4_d_theta(Scalar theta, Scalar k1, Scalar k2, Scalar k3, Scalar k4) {
    const Scalar t2 = theta * theta;
    return theta * (Scalar(1) + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
}

template <typename Scalar>
inline Scalar kb4_d_theta_derivative(Scalar theta, Scalar k1, Scalar k2, Scalar k3, Scalar k4) {
    const Scalar t2 = theta * theta;
    return Scalar(1) + t2 * (Scalar(3) * k1 + t2 * (Scalar(5) * k2 +
                                                    t2 * (Scalar(7) * k3 + t2 * Scalar(9) * k4)));
}

// Solve d(theta) = thetad. Returns theta and 1/d'(theta) at the solution.
template <typename Scalar>
inline Scalar kb4_solve_theta(Scalar thetad, Scalar k1, Scalar k2, Scalar k3, Scalar k4,
                              Scalar &dtheta_dthetad) {
    Scalar theta = thetad;
    dtheta_dthetad = Scalar(1);
    const Scalar pi = std::acos(Scalar(-1));
    const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                             std::max(Scalar(1), thetad);
    for (int iter = 0; iter < 20; ++iter) {
        const Scalar d = kb4_d_theta(theta, k1, k2, k3, k4);
        const Scalar dd = kb4_d_theta_derivative(theta, k1, k2, k3, k4);
        if (!(dd > domain_epsilon<Scalar>()) || !std::isfinite(theta)) {
            dtheta_dthetad = Scalar(0);
            return theta;
        }
        const Scalar step = (thetad - d) / dd;
        if (std::abs(thetad - d) <= tolerance) {
            break;
        }
        theta += step;
    }
    const Scalar dd = kb4_d_theta_derivative(theta, k1, k2, k3, k4);
    if (!(dd > domain_epsilon<Scalar>()) || !(theta >= Scalar(0) && theta < pi) ||
        !std::isfinite(theta) ||
        std::abs(kb4_d_theta(theta, k1, k2, k3, k4) - thetad) > tolerance) {
        dtheta_dthetad = Scalar(0);
        return theta;
    }
    dtheta_dthetad = Scalar(1) / dd;
    return theta;
}

template <typename Scalar>
inline bool project_kannala_brandt4(Scalar fx, Scalar fy, Scalar cx, Scalar cy, Scalar k1,
                                    Scalar k2, Scalar k3, Scalar k4, const Vec3<Scalar> &Xc,
                                    Vec2<Scalar> &uv, Mat23<Scalar> *J_uv_Xc = nullptr) {
    const Scalar x = Xc[0];
    const Scalar y = Xc[1];
    const Scalar z = Xc[2];
    const Scalar r2 = x * x + y * y;
    const Scalar r = std::sqrt(r2);
    const Scalar eps = domain_epsilon<Scalar>();

    if (r <= eps * std::abs(z) && z > Scalar(0)) {
        return project_pinhole(fx, fy, cx, cy, Xc, uv, J_uv_Xc);
    }
    if (!(r > Scalar(0)) || !Xc.allFinite()) {
        uv << cx, cy;
        if (J_uv_Xc) J_uv_Xc->setZero();
        return false;
    }

    const Scalar theta = std::atan2(r, z);
    const Scalar d = kb4_d_theta(theta, k1, k2, k3, k4);
    const Scalar dd = kb4_d_theta_derivative(theta, k1, k2, k3, k4);
    const Scalar inv_r = Scalar(1) / r;
    const Scalar mx = x * d * inv_r;
    const Scalar my = y * d * inv_r;
    uv[0] = fx * mx + cx;
    uv[1] = fy * my + cy;

    if (J_uv_Xc) {
        const Scalar r3 = r2 * r;
        const Scalar denom = r2 * (r2 + z * z);
        const Scalar x2 = x * x;
        const Scalar y2 = y * y;
        J_uv_Xc->setZero();
        (*J_uv_Xc)(0, 0) = fx * (dd * z * x2 / denom + d * y2 / r3);
        (*J_uv_Xc)(1, 0) = fy * (dd * z * y * x / denom - d * y * x / r3);
        (*J_uv_Xc)(0, 1) = fx * (dd * z * y * x / denom - d * y * x / r3);
        (*J_uv_Xc)(1, 1) = fy * (dd * z * y2 / denom + d * x2 / r3);
        (*J_uv_Xc)(0, 2) = -fx * dd * x / (r2 + z * z);
        (*J_uv_Xc)(1, 2) = -fy * dd * y / (r2 + z * z);
    }
    return std::isfinite(uv[0]) && std::isfinite(uv[1]);
}

template <typename Scalar>
inline bool unproject_kannala_brandt4_bearing(Scalar fx, Scalar fy, Scalar cx, Scalar cy,
                                              Scalar k1, Scalar k2, Scalar k3, Scalar k4,
                                              const Vec2<Scalar> &uv, Vec3<Scalar> &bearing,
                                              Mat32<Scalar> *J_bearing_uv = nullptr) {
    const Scalar mx = (uv[0] - cx) / fx;
    const Scalar my = (uv[1] - cy) / fy;
    const Scalar thetad = std::sqrt(mx * mx + my * my);
    const Scalar eps = domain_epsilon<Scalar>();

    if (thetad <= eps) {
        bearing << Scalar(0), Scalar(0), Scalar(1);
        if (J_bearing_uv) {
            J_bearing_uv->setZero();
            (*J_bearing_uv)(0, 0) = Scalar(1) / fx;
            (*J_bearing_uv)(1, 1) = Scalar(1) / fy;
        }
        return true;
    }

    Scalar dtheta_dthetad = Scalar(1);
    const Scalar theta = kb4_solve_theta(thetad, k1, k2, k3, k4, dtheta_dthetad);
    if (!(dtheta_dthetad > Scalar(0))) {
        bearing.setZero();
        if (J_bearing_uv) J_bearing_uv->setZero();
        return false;
    }
    const Scalar sin_theta = std::sin(theta);
    const Scalar cos_theta = std::cos(theta);
    const Scalar scaling = sin_theta / thetad;
    bearing[0] = mx * scaling;
    bearing[1] = my * scaling;
    bearing[2] = cos_theta;

    if (J_bearing_uv) {
        const Scalar dthetad_dmx = mx / thetad;
        const Scalar dthetad_dmy = my / thetad;
        const Scalar dscaling_dthetad =
            (thetad * cos_theta * dtheta_dthetad - sin_theta) / (thetad * thetad);
        const Scalar dcos_dthetad = -sin_theta * dtheta_dthetad;

        const Scalar dbx_dmx = scaling + mx * dscaling_dthetad * dthetad_dmx;
        const Scalar dbx_dmy = mx * dscaling_dthetad * dthetad_dmy;
        const Scalar dby_dmx = my * dscaling_dthetad * dthetad_dmx;
        const Scalar dby_dmy = scaling + my * dscaling_dthetad * dthetad_dmy;
        const Scalar dbz_dmx = dcos_dthetad * dthetad_dmx;
        const Scalar dbz_dmy = dcos_dthetad * dthetad_dmy;

        (*J_bearing_uv)(0, 0) = dbx_dmx / fx;
        (*J_bearing_uv)(1, 0) = dby_dmx / fx;
        (*J_bearing_uv)(2, 0) = dbz_dmx / fx;
        (*J_bearing_uv)(0, 1) = dbx_dmy / fy;
        (*J_bearing_uv)(1, 1) = dby_dmy / fy;
        (*J_bearing_uv)(2, 1) = dbz_dmy / fy;
    }
    return std::isfinite(bearing[0]) && std::isfinite(bearing[1]) && std::isfinite(bearing[2]);
}

// ---------------------------------------------------------------------------
// OpenCV rational pinhole (Radtan8):
//   k1, k2, p1, p2, k3, k4, k5, k6
// Unproject returns the point on z = 1 (the undistorted normalized coordinate),
// which matches the pinhole bearing convention on the forward hemisphere.
// ---------------------------------------------------------------------------

template <typename Scalar>
inline bool distort_radtan8(Scalar xp, Scalar yp, Scalar k1, Scalar k2, Scalar p1, Scalar p2,
                            Scalar k3, Scalar k4, Scalar k5, Scalar k6, Scalar &xpp, Scalar &ypp,
                            Mat22<Scalar> *J_dist_undist = nullptr) {
    const Scalar r2 = xp * xp + yp * yp;
    const Scalar r4 = r2 * r2;
    const Scalar r6 = r4 * r2;
    const Scalar num = Scalar(1) + k1 * r2 + k2 * r4 + k3 * r6;
    const Scalar den = Scalar(1) + k4 * r2 + k5 * r4 + k6 * r6;
    if (!(std::abs(den) > domain_epsilon<Scalar>())) {
        xpp = xp;
        ypp = yp;
        return false;
    }
    const Scalar cdist = num / den;
    const Scalar delta_x = Scalar(2) * p1 * xp * yp + p2 * (r2 + Scalar(2) * xp * xp);
    const Scalar delta_y = p1 * (r2 + Scalar(2) * yp * yp) + Scalar(2) * p2 * xp * yp;
    xpp = xp * cdist + delta_x;
    ypp = yp * cdist + delta_y;

    if (J_dist_undist) {
        const Scalar dnum_dr2 = k1 + Scalar(2) * k2 * r2 + Scalar(3) * k3 * r4;
        const Scalar dden_dr2 = k4 + Scalar(2) * k5 * r2 + Scalar(3) * k6 * r4;
        const Scalar den2 = den * den;
        const Scalar dcdist_dr2 = (dnum_dr2 * den - num * dden_dr2) / den2;
        const Scalar dcdist_dxp = dcdist_dr2 * Scalar(2) * xp;
        const Scalar dcdist_dyp = dcdist_dr2 * Scalar(2) * yp;
        const Scalar ddx_dxp = Scalar(2) * p1 * yp + Scalar(6) * p2 * xp;
        const Scalar ddx_dyp = Scalar(2) * p1 * xp + Scalar(2) * p2 * yp;
        const Scalar ddy_dxp = Scalar(2) * p1 * xp + Scalar(2) * p2 * yp;
        const Scalar ddy_dyp = Scalar(6) * p1 * yp + Scalar(2) * p2 * xp;
        (*J_dist_undist)(0, 0) = cdist + xp * dcdist_dxp + ddx_dxp;
        (*J_dist_undist)(0, 1) = xp * dcdist_dyp + ddx_dyp;
        (*J_dist_undist)(1, 0) = yp * dcdist_dxp + ddy_dxp;
        (*J_dist_undist)(1, 1) = cdist + yp * dcdist_dyp + ddy_dyp;
    }
    return true;
}

template <typename Scalar>
inline bool project_pinhole_radtan8(Scalar fx, Scalar fy, Scalar cx, Scalar cy, Scalar k1,
                                    Scalar k2, Scalar p1, Scalar p2, Scalar k3, Scalar k4,
                                    Scalar k5, Scalar k6, const Vec3<Scalar> &Xc, Vec2<Scalar> &uv,
                                    Mat23<Scalar> *J_uv_Xc = nullptr) {
    const Scalar x = Xc[0];
    const Scalar y = Xc[1];
    const Scalar z = Xc[2];
    const bool positive_z = z > domain_epsilon<Scalar>();
    const Scalar inv_z = Scalar(1) / z;
    const Scalar xp = x * inv_z;
    const Scalar yp = y * inv_z;
    Scalar xpp = Scalar(0);
    Scalar ypp = Scalar(0);
    Mat22<Scalar> Jdist;
    const bool distorted =
        distort_radtan8(xp, yp, k1, k2, p1, p2, k3, k4, k5, k6, xpp, ypp,
                        J_uv_Xc ? &Jdist : nullptr);
    uv[0] = fx * xpp + cx;
    uv[1] = fy * ypp + cy;
    if (!distorted) {
        // The rational denominator vanished, so Jdist was not written.
        if (J_uv_Xc) {
            J_uv_Xc->setZero();
        }
        return false;
    }
    if (J_uv_Xc) {
        const Scalar inv_z2 = inv_z * inv_z;
        Mat23<Scalar> Jnorm;
        Jnorm << inv_z, Scalar(0), -x * inv_z2, Scalar(0), inv_z, -y * inv_z2;
        Mat22<Scalar> focal;
        focal << fx, Scalar(0), Scalar(0), fy;
        *J_uv_Xc = focal * Jdist * Jnorm;
    }
    return positive_z && std::isfinite(uv[0]) && std::isfinite(uv[1]);
}

template <typename Scalar>
inline bool unproject_pinhole_radtan8_bearing(Scalar fx, Scalar fy, Scalar cx, Scalar cy,
                                              Scalar k1, Scalar k2, Scalar p1, Scalar p2,
                                              Scalar k3, Scalar k4, Scalar k5, Scalar k6,
                                              const Vec2<Scalar> &uv, Vec3<Scalar> &bearing,
                                              Mat32<Scalar> *J_bearing_uv = nullptr) {
    const Scalar x0 = (uv[0] - cx) / fx;
    const Scalar y0 = (uv[1] - cy) / fy;
    Scalar xp = x0;
    Scalar yp = y0;
    Mat22<Scalar> Jdist = Mat22<Scalar>::Identity();
    bool ok = true;
    for (int iter = 0; iter < 8; ++iter) {
        Scalar xpp = Scalar(0);
        Scalar ypp = Scalar(0);
        ok = distort_radtan8(xp, yp, k1, k2, p1, p2, k3, k4, k5, k6, xpp, ypp, &Jdist);
        if (!ok) {
            break;
        }
        const Vec2<Scalar> residual(xpp - x0, ypp - y0);
        if (residual.norm() < Scalar(1e-12)) {
            break;
        }
        const Scalar det = Jdist.determinant();
        if (!(std::abs(det) > domain_epsilon<Scalar>())) {
            ok = false;
            break;
        }
        const Vec2<Scalar> step = Jdist.inverse() * residual;
        xp -= step[0];
        yp -= step[1];
    }
    // Recompute the derivative at the final iterate and reject unconverged roots.
    Scalar final_x, final_y;
    ok = ok && distort_radtan8(xp, yp, k1, k2, p1, p2, k3, k4, k5, k6,
                               final_x, final_y, &Jdist);
    const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                             std::max(Scalar(1), std::max(std::abs(x0), std::abs(y0)));
    ok = ok && std::abs(final_x - x0) <= tolerance && std::abs(final_y - y0) <= tolerance;
    bearing << xp, yp, Scalar(1);
    if (J_bearing_uv) {
        const Scalar det = Jdist.determinant();
        if (std::abs(det) > domain_epsilon<Scalar>()) {
            const Mat22<Scalar> dundist_ddist = Jdist.inverse();
            J_bearing_uv->setZero();
            (*J_bearing_uv)(0, 0) = dundist_ddist(0, 0) / fx;
            (*J_bearing_uv)(1, 0) = dundist_ddist(1, 0) / fx;
            (*J_bearing_uv)(0, 1) = dundist_ddist(0, 1) / fy;
            (*J_bearing_uv)(1, 1) = dundist_ddist(1, 1) / fy;
        } else {
            J_bearing_uv->setZero();
            ok = false;
        }
    }
    return ok && std::isfinite(xp) && std::isfinite(yp);
}

// ---------------------------------------------------------------------------
// Double sphere (Usenko et al.): parameters xi, alpha.
// The bearing is the closed-form unprojection, not a z = 1 point.
// ---------------------------------------------------------------------------

template <typename Scalar>
inline bool project_double_sphere(Scalar fx, Scalar fy, Scalar cx, Scalar cy, Scalar xi,
                                  Scalar alpha, const Vec3<Scalar> &Xc, Vec2<Scalar> &uv,
                                  Mat23<Scalar> *J_uv_Xc = nullptr) {
    const Scalar x = Xc[0];
    const Scalar y = Xc[1];
    const Scalar z = Xc[2];
    const Scalar xx = x * x;
    const Scalar yy = y * y;
    const Scalar zz = z * z;
    const Scalar r2 = xx + yy;
    const Scalar d1_2 = r2 + zz;
    const Scalar d1 = std::sqrt(d1_2);
    if (!(d1 > domain_epsilon<Scalar>())) {
        uv << cx, cy;
        if (J_uv_Xc) {
            J_uv_Xc->setZero();
        }
        return false;
    }

    const Scalar w1 = alpha > Scalar(0.5) ? (Scalar(1) - alpha) / alpha : alpha / (Scalar(1) - alpha);
    const Scalar w2 = (w1 + xi) / std::sqrt(Scalar(2) * w1 * xi + xi * xi + Scalar(1));
    const bool valid = z > -w2 * d1;

    const Scalar k = xi * d1 + z;
    const Scalar d2 = std::sqrt(r2 + k * k);
    const Scalar norm = alpha * d2 + (Scalar(1) - alpha) * k;
    if (!(std::abs(norm) > domain_epsilon<Scalar>())) {
        uv << cx, cy;
        if (J_uv_Xc) {
            J_uv_Xc->setZero();
        }
        return false;
    }
    const Scalar mx = x / norm;
    const Scalar my = y / norm;
    uv[0] = fx * mx + cx;
    uv[1] = fy * my + cy;

    if (J_uv_Xc) {
        const Scalar norm2 = norm * norm;
        const Scalar xy = x * y;
        const Scalar tt2 = xi * z / d1 + Scalar(1);
        const Scalar d_norm_d_r2 =
            (xi * (Scalar(1) - alpha) / d1 + alpha * (xi * k / d1 + Scalar(1)) / d2) / norm2;
        const Scalar tmp2 =
            ((Scalar(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;
        J_uv_Xc->setZero();
        (*J_uv_Xc)(0, 0) = fx * (Scalar(1) / norm - xx * d_norm_d_r2);
        (*J_uv_Xc)(1, 0) = -fy * xy * d_norm_d_r2;
        (*J_uv_Xc)(0, 1) = -fx * xy * d_norm_d_r2;
        (*J_uv_Xc)(1, 1) = fy * (Scalar(1) / norm - yy * d_norm_d_r2);
        (*J_uv_Xc)(0, 2) = -fx * x * tmp2;
        (*J_uv_Xc)(1, 2) = -fy * y * tmp2;
    }
    return valid && std::isfinite(uv[0]) && std::isfinite(uv[1]);
}

template <typename Scalar>
inline bool unproject_double_sphere_bearing(Scalar fx, Scalar fy, Scalar cx, Scalar cy, Scalar xi,
                                            Scalar alpha, const Vec2<Scalar> &uv,
                                            Vec3<Scalar> &bearing,
                                            Mat32<Scalar> *J_bearing_uv = nullptr) {
    const Scalar mx = (uv[0] - cx) / fx;
    const Scalar my = (uv[1] - cy) / fy;
    const Scalar r2 = mx * mx + my * my;
    const Scalar a = alpha;
    const Scalar a2 = a * a;
    const Scalar xi2 = xi * xi;
    const Scalar radicand = Scalar(1) - (Scalar(2) * a - Scalar(1)) * r2;
    const bool inside = !(a > Scalar(0.5) && r2 >= Scalar(1) / (Scalar(2) * a - Scalar(1)));
    if (!(radicand > Scalar(0))) {
        bearing.setZero();
        if (J_bearing_uv) {
            J_bearing_uv->setZero();
        }
        return false;
    }
    const Scalar sqrt2 = std::sqrt(radicand);
    const Scalar norm2 = a * sqrt2 + (Scalar(1) - a);
    const Scalar mz = (Scalar(1) - a2 * r2) / norm2;
    const Scalar mz2 = mz * mz;
    const Scalar norm1 = mz2 + r2;
    const Scalar inside_sqrt = mz2 + (Scalar(1) - xi2) * r2;
    if (!(inside_sqrt > Scalar(0)) || !(std::abs(norm1) > domain_epsilon<Scalar>())) {
        bearing.setZero();
        if (J_bearing_uv) {
            J_bearing_uv->setZero();
        }
        return false;
    }
    const Scalar sqrt1 = std::sqrt(inside_sqrt);
    const Scalar k = (mz * xi + sqrt1) / norm1;
    bearing[0] = k * mx;
    bearing[1] = k * my;
    bearing[2] = k * mz - xi;

    if (J_bearing_uv) {
        const Scalar ds_dr2 = -(Scalar(2) * a - Scalar(1)) / (Scalar(2) * sqrt2);
        const Scalar dnorm2_dr2 = a * ds_dr2;
        const Scalar num = Scalar(1) - a2 * r2;
        const Scalar dmz_dr2 = (-a2 * norm2 - num * dnorm2_dr2) / (norm2 * norm2);
        const Scalar dsqrt1_dr2 =
            (Scalar(2) * mz * dmz_dr2 + (Scalar(1) - xi2)) / (Scalar(2) * sqrt1);
        const Scalar dnorm1_dr2 = Scalar(2) * mz * dmz_dr2 + Scalar(1);
        const Scalar dn1_dr2 = xi * dmz_dr2 + dsqrt1_dr2;
        const Scalar n1 = mz * xi + sqrt1;
        const Scalar norm1_2 = norm1 * norm1;
        const Scalar dk_dr2 = (dn1_dr2 * norm1 - n1 * dnorm1_dr2) / norm1_2;

        const Scalar dk_dmx = dk_dr2 * Scalar(2) * mx;
        const Scalar dk_dmy = dk_dr2 * Scalar(2) * my;
        const Scalar dmz_dmx = dmz_dr2 * Scalar(2) * mx;
        const Scalar dmz_dmy = dmz_dr2 * Scalar(2) * my;

        (*J_bearing_uv)(0, 0) = (mx * dk_dmx + k) / fx;
        (*J_bearing_uv)(1, 0) = (my * dk_dmx) / fx;
        (*J_bearing_uv)(2, 0) = (mz * dk_dmx + k * dmz_dmx) / fx;
        (*J_bearing_uv)(0, 1) = (mx * dk_dmy) / fy;
        (*J_bearing_uv)(1, 1) = (my * dk_dmy + k) / fy;
        (*J_bearing_uv)(2, 1) = (mz * dk_dmy + k * dmz_dmy) / fy;
    }
    return inside && std::isfinite(bearing[0]) && std::isfinite(bearing[1]) &&
           std::isfinite(bearing[2]);
}

// ---------------------------------------------------------------------------
// Factor Jacobians. J_uv_Xc is d(pi)/d(Xc). These blocks are derivatives of the
// g2o reprojection error (measurement - pi). gtsam factors use pi - measurement
// and must not apply this minus sign.
// ---------------------------------------------------------------------------

template <typename Scalar>
inline void g2o_left_se3_jacobian(const Vec3<Scalar> &Xc, Mat36<Scalar> &dXc_dxi) {
    const Scalar x = Xc[0];
    const Scalar y = Xc[1];
    const Scalar z = Xc[2];
    dXc_dxi << Scalar(0), z, -y, Scalar(1), Scalar(0), Scalar(0), -z, Scalar(0), x, Scalar(0),
        Scalar(1), Scalar(0), y, -x, Scalar(0), Scalar(0), Scalar(0), Scalar(1);
}

template <typename Scalar>
inline void reprojection_error_jacobians(const Mat23<Scalar> &J_uv_Xc,
                                         const Mat36<Scalar> &dXc_dpose,
                                         const Mat33<Scalar> &dXc_dpoint, Mat26<Scalar> &J_error_pose,
                                         Mat23<Scalar> &J_error_point) {
    J_error_pose = -J_uv_Xc * dXc_dpose;
    J_error_point = -J_uv_Xc * dXc_dpoint;
}

// g2o VertexSE3Expmap uses a left perturbation exp(xi) * T.
// dXc/dpoint is Rcw when Xc = Rcw * Xw + tcw.
template <typename Scalar>
inline void g2o_reprojection_error_jacobians(const Mat23<Scalar> &J_uv_Xc, const Vec3<Scalar> &Xc,
                                             const Mat33<Scalar> &Rcw, Mat26<Scalar> &J_error_pose,
                                             Mat23<Scalar> &J_error_point) {
    Mat36<Scalar> dXc_dxi;
    g2o_left_se3_jacobian(Xc, dXc_dxi);
    reprojection_error_jacobians(J_uv_Xc, dXc_dxi, Rcw, J_error_pose, J_error_point);
}

} // namespace camera_model
} // namespace pyslam
