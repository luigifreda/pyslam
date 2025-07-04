#pragma once

#include <pangolin/display/opengl_render_state.h>
#include <pangolin/display/viewport.h>
#include <pangolin/gl/gldraw.h>

#include <pangolin/scene/renderable.h>
#include <pangolin/scene/interactive_index.h>

#ifdef HAVE_EIGEN
#  include <Eigen/Geometry>
#endif

namespace pangolin {

struct Axis : public Renderable, public Interactive
{
    Axis()
        : axis_length(1.0),
          label_x(InteractiveIndex::I().Store(this)),
          label_y(InteractiveIndex::I().Store(this)),
          label_z(InteractiveIndex::I().Store(this))
    {
    }

    void Render(const RenderParams&) override {
        glColor4f(1,0,0,1);
        glPushName(label_x.Id());
        glDrawLine(0,0,0, axis_length,0,0);
        glPopName();

        glColor4f(0,1,0,1);
        glPushName(label_y.Id());
        glDrawLine(0,0,0, 0,axis_length,0);
        glPopName();

        glColor4f(0,0,1,1);
        glPushName(label_z.Id());
        glDrawLine(0,0,0, 0,0,axis_length);
        glPopName();
    }

    bool Mouse(
        int button,
        const GLprecision /*win*/[3], const GLprecision /*obj*/[3], const GLprecision /*normal*/[3],
        bool /*pressed*/, int button_state, int pickId
    ) override
    {
        PANGOLIN_UNUSED(button);
        PANGOLIN_UNUSED(button_state);
        PANGOLIN_UNUSED(pickId);

#ifdef HAVE_EIGEN
        if((button == MouseWheelUp || button == MouseWheelDown) ) {
            float scale = (button == MouseWheelUp) ? 0.01f : -0.01f;
            if(button_state & KeyModifierShift) scale /= 10;

            Eigen::Vector3d rot = Eigen::Vector3d::Zero();
            Eigen::Vector3d xyz = Eigen::Vector3d::Zero();


            if(button_state & KeyModifierCtrl) {
                // rotate
                if(pickId == label_x.Id()) {
                    rot << 1,0,0;
                }else if(pickId == label_y.Id()) {
                    rot << 0,1,0;
                }else if(pickId == label_z.Id()) {
                    rot << 0,0,1;
                }else{
                    return false;
                }
            }else if(button_state & KeyModifierShift){
                // translate
                if(pickId == label_x.Id()) {
                    xyz << 1,0,0;
                }else if(pickId == label_y.Id()) {
                    xyz << 0,1,0;
                }else if(pickId == label_z.Id()) {
                    xyz << 0,0,1;
                }else{
                    return false;
                }
            }else{
                return false;
            }

            // old from new
            Eigen::Matrix<double,4,4> T_on = Eigen::Matrix<double,4,4>::Identity();
            T_on.block<3,3>(0,0) = Eigen::AngleAxis<double>(scale,rot).toRotationMatrix();
            T_on.block<3,1>(0,3) = scale*xyz;

            // Update
            T_pc = (ToEigen<double>(T_pc) * T_on.inverse()).eval();

            return true;
        }
#endif // HAVE_EIGEN

        return false;
    }

    virtual bool MouseMotion(
        const GLprecision /*win*/[3], const GLprecision /*obj*/[3], const GLprecision /*normal*/[3],
        int /*button_state*/, int /*pickId*/
    ) override
    {
        return false;
    }

    float axis_length;
    const InteractiveIndex::Token label_x;
    const InteractiveIndex::Token label_y;
    const InteractiveIndex::Token label_z;
};

}
