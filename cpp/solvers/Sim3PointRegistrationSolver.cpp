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

#include "Sim3PointRegistrationSolver.h"
#include "Random.h"

#include <cmath>
#include <opencv2/core/core.hpp>
#include <vector>

#include <Eigen/Eigenvalues>

using namespace std;

namespace utils {

static constexpr bool kVerbose = false;

Sim3PointRegistrationSolver::Sim3PointRegistrationSolver(
    const Sim3PointRegistrationSolverInput &input)
    : mnIterations(0), mnBestInliers(0) {
    assert(input.mvX3Dw1.size() == input.mvX3Dw2.size());

    mN = input.mvX3Dw1.size(); // number of correspondences

    mvnIndices1.reserve(mN);
    mvX3Dw1.reserve(mN);
    mvX3Dw2.reserve(mN);

    if (input.mSigma2 > 0) {
        mSigma2 = input.mSigma2;
    } else {
        mSigma2 = kSigma2;
    }
    // 7.81 value of the inverse chi-squared cumulative distribution for 3 DOFs and alpha=0.95
    mThChi2 = 7.81 * mSigma2;

    mbFixScale = input.bFixScale;

    if (kVerbose) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Sim3PointRegistrationSolver input:" << std::endl;
        std::cout << "Number of correspondences: " << mN << std::endl;
        std::cout << "Fix scale: " << mbFixScale << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    mvAllIndices.reserve(mN);

    for (int i1 = 0; i1 < mN; i1++) {
        mvX3Dw1.push_back(input.mvX3Dw1[i1]);
        mvX3Dw2.push_back(input.mvX3Dw2[i1]);

        mvnIndices1.push_back(i1);
        mvAllIndices.push_back(i1);
    }

    SetRansacParameters();
}

void Sim3PointRegistrationSolver::SetRansacParameters(double probability, int minInliers,
                                                      int maxIterations) {
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;

    assert(mN == mvX3Dw1.size());
    assert(mN == mvX3Dw2.size());

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

Eigen::Matrix4f Sim3PointRegistrationSolver::iterate(const int nIterations, bool &bNoMore,
                                                     vector<uint8_t> &vbInliers, int &nInliers,
                                                     bool &bConverged) {
    bNoMore = false;
    bConverged = false;
    vbInliers = std::vector<uint8_t>(mN, 0);
    nInliers = 0;

    if (mN < mRansacMinInliers) {
        if (kVerbose)
            std::cout << "[Sim3PointRegistrationSolver] Not enough correspondences" << std::endl;
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
            std::cout << "[Sim3PointRegistrationSolver] Iteration " << mnIterations << " of "
                      << nIterations << std::endl;
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for (short i = 0; i < 3; ++i) {
            int randi = Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            P3Dc1i.col(i) = mvX3Dw1[idx];
            P3Dc2i.col(i) = mvX3Dw2[idx];

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

Eigen::Matrix4f Sim3PointRegistrationSolver::find(vector<uint8_t> &vbInliers12, int &nInliers,
                                                  bool &bConverged) {
    bool bFlag;
    return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers, bConverged);
}

void Sim3PointRegistrationSolver::ComputeCentroid(Eigen::Matrix3f &P, Eigen::Matrix3f &Pr,
                                                  Eigen::Vector3f &C) {
    C = P.rowwise().sum();
    C = C / P.cols();
    for (int i = 0; i < P.cols(); i++)
        Pr.col(i) = P.col(i) - C;
}

bool Sim3PointRegistrationSolver::ComputeSim3(Eigen::Matrix3f &P1, Eigen::Matrix3f &P2) {
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

void Sim3PointRegistrationSolver::CheckInliers() {
    mnInliersi = 0;

    std::vector<float> vErrors(mN);
    std::vector<float> vADs(mN); // vector of absolute deviations

    for (size_t ii = 0; ii < mN; ii++) {
        const float error = (ms12i * mR12i * mvX3Dw2[ii] + mt12i - mvX3Dw1[ii]).norm();
        vErrors[ii] = error;

        if (error * error < mThChi2) {
            mvbInliersi[ii] = true;
            mnInliersi++;
        } else
            mvbInliersi[ii] = false;
    }

#if 0
    // Update sigma
    //if(kVerbose)
    {
        std::cout << "mSigma (before): " << std::sqrt(mSigma2) << std::endl;
    }

    // compute the robust sigma via MAD
    std::sort(vErrors.begin(), vErrors.end());
    size_t middleIdx = vErrors.size()/2;
    const float median = (mN % 2 == 0) ? (vErrors[middleIdx-1] + vErrors[middleIdx]) / 2 : vErrors[middleIdx];
    for (size_t ii=0; ii<mN; ii++)
    {
        vADs[ii] = fabs(vErrors[ii] - median);
    }
    std::sort(vADs.begin(), vADs.end());
    const float medianAD = (mN % 2 == 0) ? (vADs[middleIdx-1] + vADs[middleIdx]) / 2 : vADs[middleIdx];
    const float sigma = 1.4826 * medianAD;

    mSigma2 = sigma*sigma;
    mThChi2 = 7.81*mSigma2;

    //if(kVerbose)
    {
        std::cout << "mSigma (after): " << sigma << std::endl;
    }
#endif
}

Eigen::Matrix4f Sim3PointRegistrationSolver::GetEstimatedTransformation() { return mBestT12; }

Eigen::Matrix3f Sim3PointRegistrationSolver::GetEstimatedRotation() { return mBestRotation; }

Eigen::Vector3f Sim3PointRegistrationSolver::GetEstimatedTranslation() { return mBestTranslation; }

float Sim3PointRegistrationSolver::GetEstimatedScale() { return mBestScale; }

float Sim3PointRegistrationSolver::Compute3dRegistrationError() {
    const auto &R12 = mBestRotation;
    const auto &t12 = mBestTranslation;
    const auto &s12 = mBestScale;

    // compute the registration error (including all, not just inliers)
    const size_t N = mvX3Dw1.size();
    float error = 0.0f;
    for (size_t ii = 0; ii < N; ii++) {
        error += (s12 * R12 * mvX3Dw2[ii] + t12 - mvX3Dw1[ii]).norm();
    }
    return error / N;
}

} // namespace utils
