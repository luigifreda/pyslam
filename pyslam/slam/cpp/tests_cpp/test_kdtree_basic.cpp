#include "ckdtree_eigen.h"
#include "eigen_aliases.h"

#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace pyslam;

int main(int argc, char **argv) {

    // Fixed 2D
    MatNx2<double> P2(5, 2);
    P2 << 0, 0, 1, 0, 0, 1, 1, 1, 2, 2;
    CKDTreeEigen<double, 2> kd2{Eigen::Ref<const MatNx2<double>>(P2)};

    // Fixed 3D
    MatNx3<float> P3(3, 3);
    P3 << 0, 0, 0, 1, 1, 1, 2, 2, 2;
    CKDTreeEigen<float, 3> kd3{Eigen::Ref<const MatNx3<float>>(P3)};

    // Dynamic D (MatNxM)
    MatNxM<double> PM(4, 5); // N=4, D=5
    PM.setRandom();
    CKDTreeEigenDyn<double, size_t> kdm{Eigen::Ref<const MatNxM<double>>(PM)};

    // Queries
    double q2[2] = {0.9, 0.9};
    auto [d2, i2] = kd2.query(q2, 1);

    Eigen::Vector3f x3(0.9f, 0.9f, 0.9f);
    auto [d3, i3] = kd3.query(x3, 1);

    Eigen::VectorXd xm = Eigen::VectorXd::Random(PM.cols());
    auto [dm, im] = kdm.query(xm, 3);

    std::cout << "CKDTree tests ok" << std::endl;
    return 0;
}