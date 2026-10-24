#include "geom_2views.h"
#include "camera.h"
#include "frame.h"
#include "smart_pointers.h"

namespace pyslam {

namespace geom_2views {
/**
 * Compute fundamental matrix F12 and infinite homography H21 from two frames
 * This is the main interface function
 */
std::pair<Mat3d, Mat3d> computeF12(const FramePtr &f1, const FramePtr &f2) {

    if (!f1 || !f2) {
        throw std::invalid_argument("Frames cannot be null");
        return std::make_pair(Mat3d::Zero(), Mat3d::Zero());
    }

    // Get camera poses
    const auto f1_Tcw = f1->Tcw();
    const auto f2_Tcw = f2->Tcw();
    const auto camera1 = f1->camera;
    const auto camera2 = f2->camera;

    if (!camera1 || !camera2) {
        throw std::invalid_argument("Cameras cannot be null");
        return std::make_pair(Mat3d::Zero(), Mat3d::Zero());
    }

    // Extract rotation and translation
    Mat3d R1w = f1_Tcw.block<3, 3>(0, 0);
    Vec3d t1w = f1_Tcw.block<3, 1>(0, 3);
    Mat3d R2w = f2_Tcw.block<3, 3>(0, 0);
    Vec3d t2w = f2_Tcw.block<3, 1>(0, 3);

    // Call optimized computation
    return computeF12(R1w, t1w, R2w, t2w, camera1->Kinv, camera2->K, camera2->Kinv);
}

} // namespace geom_2views
} // namespace pyslam