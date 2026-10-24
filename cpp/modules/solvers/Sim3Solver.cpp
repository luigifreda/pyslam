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

/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel
 * and Juan D. Tardós, University of Zaragoza. Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M.
 * Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "Sim3Solver.h"
#include "Random.h"

#include <cmath>
#include <opencv2/core/core.hpp>
#include <vector>

#include <Eigen/Eigenvalues>

using namespace std;

namespace utils {

static constexpr bool kVerbose = false;

Sim3Solver::Sim3Solver(const Sim3SolverInput &input) : mnIterations(0), mnBestInliers(0) {
    assert(input.mvX3Dw1.size() == input.mvX3Dw2.size());
    assert(input.mvX3Dw1.size() == input.mvSigmaSquare1.size());
    assert(input.mvX3Dw1.size() == input.mvSigmaSquare2.size());

    mN = input.mvX3Dw1.size(); // number of correspondences

    mvnIndices1.reserve(mN);
    mvX3Dc1.reserve(mN);
    mvX3Dc2.reserve(mN);

    mK1 = input.K1;
    mK2 = input.K2;
    mbFixScale = input.bFixScale;

    const Eigen::Matrix3f &Rcw1 = input.Rcw1;
    const Eigen::Vector3f &tcw1 = input.tcw1;
    const Eigen::Matrix3f &Rcw2 = input.Rcw2;
    const Eigen::Vector3f &tcw2 = input.tcw2;

    if (kVerbose) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Sim3Solver input:" << std::endl;
        std::cout << "Number of correspondences: " << mN << std::endl;
        std::cout << "Fix scale: " << mbFixScale << std::endl;
        std::cout << "K1: " << mK1 << std::endl;
        std::cout << "K2: " << mK2 << std::endl;
        std::cout << "Rcw1: " << Rcw1 << std::endl;
        std::cout << "tcw1: " << tcw1 << std::endl;
        std::cout << "Rcw2: " << Rcw2 << std::endl;
        std::cout << "tcw2: " << tcw2 << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    mvAllIndices.reserve(mN);

    for (int i1 = 0; i1 < mN; i1++) {
        // 9.210 is 2-DOFs chi-squared value with alpha=0.99
        mvnMaxError1.push_back(9.210 * input.mvSigmaSquare1[i1]);
        mvnMaxError2.push_back(9.210 * input.mvSigmaSquare2[i1]);

        mvX3Dc1.push_back(Rcw1 * input.mvX3Dw1[i1] + tcw1);
        mvX3Dc2.push_back(Rcw2 * input.mvX3Dw2[i1] + tcw2);

        mvnIndices1.push_back(i1);
        mvAllIndices.push_back(i1);
    }

    FromCameraToImage(mvX3Dc1, mK1, mvP1im1);
    FromCameraToImage(mvX3Dc2, mK2, mvP2im2);

    SetRansacParameters();
}

Sim3Solver::Sim3Solver(const Sim3SolverInput2 &input) : mnIterations(0), mnBestInliers(0) {
    assert(input.mvX3Dc1.size() == input.mvX3Dc2.size());
    assert(input.mvX3Dc1.size() == input.mvSigmaSquare1.size());
    assert(input.mvX3Dc1.size() == input.mvSigmaSquare2.size());

    mN = input.mvX3Dc1.size(); // number of correspondences

    mvnIndices1.reserve(mN);
    mvX3Dc1 = input.mvX3Dc1;
    mvX3Dc2 = input.mvX3Dc2;

    mK1 = input.K1;
    mK2 = input.K2;
    mbFixScale = input.bFixScale;

    if (kVerbose) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Sim3Solver input:" << std::endl;
        std::cout << "Number of correspondences: " << mN << std::endl;
        std::cout << "Fix scale: " << mbFixScale << std::endl;
        std::cout << "K1: " << mK1 << std::endl;
        std::cout << "K2: " << mK2 << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    mvAllIndices.reserve(mN);

    for (int i1 = 0; i1 < mN; i1++) {
        // 9.210 is 2-DOFs chi-squared value with alpha=0.99
        mvnMaxError1.push_back(9.210 * input.mvSigmaSquare1[i1]);
        mvnMaxError2.push_back(9.210 * input.mvSigmaSquare2[i1]);

        mvnIndices1.push_back(i1);
        mvAllIndices.push_back(i1);
    }

    FromCameraToImage(mvX3Dc1, mK1, mvP1im1);
    FromCameraToImage(mvX3Dc2, mK2, mvP2im2);

    SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations) {
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;

    assert(mN == mvX3Dc1.size());
    assert(mN == mvX3Dc2.size());

    mvbInliersi.resize(mN);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers / mN;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (mRansacMinInliers == mN)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

    mRansacMaxIts = std::max(1, std::min(nIterations, mRansacMaxIts));

    mnIterations = 0;
}

Eigen::Matrix4f Sim3Solver::iterate(const int nIterations, bool &bNoMore,
                                    vector<uint8_t> &vbInliers, int &nInliers, bool &bConverged) {
    bNoMore = false;
    bConverged = false;
    vbInliers = std::vector<uint8_t>(mN, 0);
    nInliers = 0;

    if (mN < mRansacMinInliers) {
        if (kVerbose)
            std::cout << "[Sim3Solver] Not enough correspondences" << std::endl;
        bNoMore = true;
        return Eigen::Matrix4f::Identity();
    }

    vector<size_t> vAvailableIndices;

    Eigen::Matrix3f P3Dc1i;
    Eigen::Matrix3f P3Dc2i;

    int nCurrentIterations = 0;

    Eigen::Matrix4f bestSim3;

    while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations) {
        if (kVerbose)
            std::cout << "[Sim3Solver] Iteration " << mnIterations << " of " << nIterations
                      << std::endl;
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for (short i = 0; i < 3; ++i) {
            int randi = Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            P3Dc1i.col(i) = mvX3Dc1[idx];
            P3Dc2i.col(i) = mvX3Dc2[idx];

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        if (!ComputeSim3(P3Dc1i, P3Dc2i))
            continue;

        CheckInliers();

        if (kVerbose) {
            std::cout << "mnInliersi: " << mnInliersi << std::endl;
            std::cout << "mT12i: " << mT12i << std::endl;
            std::cout << "mR12i: " << mR12i << std::endl;
            std::cout << "mt12i: " << mt12i << std::endl;
            std::cout << "ms12i: " << ms12i << std::endl;
        }

        if (mnInliersi >= mnBestInliers) {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i;
            mBestRotation = mR12i;
            mBestTranslation = mt12i;
            mBestScale = ms12i;

            if (mnInliersi > mRansacMinInliers) {
                nInliers = mnInliersi;
                for (int i = 0; i < mN; i++)
                    if (mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = 1;
                bConverged = true;
                return mBestT12;
            } else {
                bestSim3 = mBestT12;
            }
        }
    }

    if (mnIterations >= mRansacMaxIts)
        bNoMore = true;

    // Even if convergence failed, report the best number of inliers found
    // This helps with debugging why convergence failed
    if (mnBestInliers > 0) {
        nInliers = mnBestInliers;
        for (int i = 0; i < mN; i++)
            if (mvbBestInliers[i])
                vbInliers[mvnIndices1[i]] = 1;
    }

    return bestSim3;
}

Eigen::Matrix4f Sim3Solver::find(vector<uint8_t> &vbInliers12, int &nInliers, bool &bConverged) {
    bool bFlag;
    return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers, bConverged);
}

void Sim3Solver::ComputeCentroid(Eigen::Matrix3f &P, Eigen::Matrix3f &Pr, Eigen::Vector3f &C) {
    C = P.rowwise().sum();
    C = C / P.cols();
    for (int i = 0; i < P.cols(); i++)
        Pr.col(i) = P.col(i) - C;
}

bool Sim3Solver::ComputeSim3(Eigen::Matrix3f &P1, Eigen::Matrix3f &P2) {
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientation using unit quaternions

    // Step 1: Centroid and relative coordinates

    Eigen::Matrix3f Pr1; // Relative coordinates to centroid (set 1)
    Eigen::Matrix3f Pr2; // Relative coordinates to centroid (set 2)
    Eigen::Vector3f O1;  // Centroid of P1
    Eigen::Vector3f O2;  // Centroid of P2

    ComputeCentroid(P1, Pr1, O1);
    ComputeCentroid(P2, Pr2, O2);

    // Step 2: Compute M matrix

    Eigen::Matrix3f M = Pr2 * Pr1.transpose();

    // Step 3: Compute N matrix
    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    Eigen::Matrix4f N;

    N11 = M(0, 0) + M(1, 1) + M(2, 2);
    N12 = M(1, 2) - M(2, 1);
    N13 = M(2, 0) - M(0, 2);
    N14 = M(0, 1) - M(1, 0);
    N22 = M(0, 0) - M(1, 1) - M(2, 2);
    N23 = M(0, 1) + M(1, 0);
    N24 = M(2, 0) + M(0, 2);
    N33 = -M(0, 0) + M(1, 1) - M(2, 2);
    N34 = M(1, 2) + M(2, 1);
    N44 = -M(0, 0) - M(1, 1) + M(2, 2);

    N << N11, N12, N13, N14, N12, N22, N23, N24, N13, N23, N33, N34, N14, N24, N34, N44;

    // Step 4: Eigenvector of the highest eigenvalue
    Eigen::EigenSolver<Eigen::Matrix4f> eigSolver;
    eigSolver.compute(N);

    Eigen::Vector4f eval = eigSolver.eigenvalues().real();
    Eigen::Matrix4f evec =
        eigSolver.eigenvectors().real(); // evec[0] is the quaternion of the desired rotation

    int maxIndex; // should be zero
    eval.maxCoeff(&maxIndex);

    Eigen::Vector4f q = evec.col(maxIndex); // Quaternion (w, x, y, z)
    Eigen::Quaternionf quat(q(0), q(1), q(2), q(3));
    mR12i = quat.toRotationMatrix();

    // Step 5: Rotate set 2
    Eigen::Matrix3f P3 = mR12i * Pr2;

    // Step 6: Scale

    if (!mbFixScale) {

        double nom = (Pr1.array() * P3.array()).sum();
        Eigen::Array<float, 3, 3> aux_P3;
        aux_P3 = P3.array() * P3.array();
        double den = aux_P3.sum();

        ms12i = nom / den;
    } else
        ms12i = 1.0f;

    // Step 7: Translation
    mt12i = O1 - ms12i * mR12i * O2;

    // Step 8: Transformation

    // Step 8.1 T12
    mT12i.setIdentity();

    Eigen::Matrix3f sR = ms12i * mR12i;
    mT12i.block<3, 3>(0, 0) = sR;
    mT12i.block<3, 1>(0, 3) = mt12i;

    // Step 8.2 T21
    mT21i.setIdentity();
    Eigen::Matrix3f sRinv = (1.0 / ms12i) * mR12i.transpose();

    // sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    mT21i.block<3, 3>(0, 0) = sRinv;

    Eigen::Vector3f tinv = -sRinv * mt12i;
    mT21i.block<3, 1>(0, 3) = tinv;
    return true;
}

void Sim3Solver::CheckInliers() {
    vector<Eigen::Vector2f> vP1im2, vP2im1;
    vector<float> depths1, depths2;
    Project(mvX3Dc2, mT12i, mK1, vP2im1, depths1);
    Project(mvX3Dc1, mT21i, mK2, vP1im2, depths2);

    mnInliersi = 0;

    for (size_t i = 0; i < mvP1im1.size(); i++) {
        Eigen::Vector2f dist1 = mvP1im1[i] - vP2im1[i];
        Eigen::Vector2f dist2 = vP1im2[i] - mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i] && depths1[i] > 0 && depths2[i] > 0) {
            mvbInliersi[i] = true;
            mnInliersi++;
        } else
            mvbInliersi[i] = false;
    }
}

Eigen::Matrix4f Sim3Solver::GetEstimatedTransformation() { return mBestT12; }

Eigen::Matrix3f Sim3Solver::GetEstimatedRotation() { return mBestRotation; }

Eigen::Vector3f Sim3Solver::GetEstimatedTranslation() { return mBestTranslation; }

float Sim3Solver::GetEstimatedScale() { return mBestScale; }

float Sim3Solver::Compute3dRegistrationError() {
    const auto &R12 = mBestRotation;
    const auto &t12 = mBestTranslation;
    const auto &s12 = mBestScale;

    // compute the registration error (including all, not just inliers)
    const size_t N = mvX3Dc1.size();
    float error = 0.0f;
    for (size_t ii = 0; ii < N; ii++) {
        error += (s12 * R12 * mvX3Dc2[ii] + t12 - mvX3Dc1[ii]).norm();
    }
    return error / N;
}

void Sim3Solver::Project(const vector<Eigen::Vector3f> &vP3Dw, const Eigen::Matrix4f &Tcw,
                         const Eigen::Matrix3f &K, vector<Eigen::Vector2f> &vP2D,
                         std::vector<float> &depths) {
    Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
    Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
    const float &fx = K(0, 0);
    const float &fy = K(1, 1);
    const float &cx = K(0, 2);
    const float &cy = K(1, 2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());
    depths.clear();
    depths.reserve(vP3Dw.size());

    for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++) {
        const Eigen::Vector3f P3Dc = Rcw * vP3Dw[i] + tcw;
        const float invz = 1.0f / P3Dc(2);
        const Eigen::Vector2f pt2D =
            Eigen::Vector2f(fx * P3Dc(0) * invz + cx, fy * P3Dc(1) * invz + cy);
        vP2D.push_back(pt2D);
        depths.push_back(P3Dc(2));
    }
}

void Sim3Solver::FromCameraToImage(const vector<Eigen::Vector3f> &vP3Dc, const Eigen::Matrix3f &K,
                                   vector<Eigen::Vector2f> &vP2D) {
    const float &fx = K(0, 0);
    const float &fy = K(1, 1);
    const float &cx = K(0, 2);
    const float &cy = K(1, 2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++) {
        const float invz = 1.0f / vP3Dc[i](2);
        const Eigen::Vector2f pt2D =
            Eigen::Vector2f(fx * vP3Dc[i](0) * invz + cx, fy * vP3Dc[i](1) * invz + cy);
        vP2D.push_back(pt2D);
    }
}

} // namespace utils
