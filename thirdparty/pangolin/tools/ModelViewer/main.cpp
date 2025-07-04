#include <thread>
#include <future>
#include <queue>

#include <pangolin/pangolin.h>
#include <pangolin/geometry/geometry.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>

#include <pangolin/utils/file_utils.h>

#include <pangolin/geometry/geometry_ply.h>
#include <pangolin/geometry/glgeometry.h>

#include <pangolin/utils/argagg.hpp>

#include "shader.h"
#include "rendertree.h"
#include "util.h"

#include <Eigen/SVD>
#include <Eigen/Geometry>

int main( int argc, char** argv )
{
    const float w = 640.0f;
    const float h = 480.0f;
    const float f = 300.0f;

    using namespace pangolin;

    argagg::parser argparser {{
        { "help", {"-h", "--help"}, "Print usage information and exit.", 0},
        { "model", {"-m","--model","--mesh"}, "3D Model to load (obj or ply)", 1},
        { "matcap", {"--matcap"}, "Matcap (material capture) images to load for shading", 1},
        { "envmap", {"--envmap","-e"}, "Equirect environment map for skybox", 1},
        { "mode", {"--mode"}, "Render mode to use {show_uv, show_texture, show_color, show_normal, show_matcap, show_vertex}", 1},
        { "bounds", {"--aabb"}, "Show axis-aligned bounding-box", 0},
        { "cull_backfaces", {"--cull"}, "Enable backface culling", 0},
        { "spin", {"--spin"}, "Spin models around an axis {none, negx, x, negy, y, negz, z}", 1},
    }};

    argagg::parser_results args = argparser.parse(argc, argv);
    if ( (bool)args["help"] || !args.has_option("model")) {
        std::cerr << "usage: ModelViewer [options]" << std::endl
                  << argparser << std::endl;
        return 0;
    }

    // Options
    enum class RenderMode { uv=0, tex, color, normal, matcap, vertex, num_modes };
    const std::string mode_names[] = {"SHOW_UV", "SHOW_TEXTURE", "SHOW_COLOR", "SHOW_NORMAL", "SHOW_MATCAP", "SHOW_VERTEX"};
    const std::string spin_names[] = {"NONE", "NEGX", "X", "NEGY", "Y", "NEGZ", "Z"};
    const char mode_key[] = {'u','t','c','n','m','v'};
    RenderMode current_mode = RenderMode::normal;
    for(int i=0; i < (int)RenderMode::num_modes; ++i)
        if(pangolin::ToUpperCopy(args["mode"].as<std::string>("SHOW_NORMAL")) == mode_names[i])
            current_mode = RenderMode(i);
    pangolin::AxisDirection spin_direction = pangolin::AxisNone;
    for(int i=0; i <= (int)pangolin::AxisZ; ++i)
        if(pangolin::ToUpperCopy(args["spin"].as<std::string>("none")) == spin_names[i])
            spin_direction = pangolin::AxisDirection(i);

    bool show_bounds = args.has_option("bounds");
    bool show_axis = args.has_option("show_axis");
    bool show_x0 = args.has_option("show_x0");
    bool show_y0 = args.has_option("show_y0");
    bool show_z0 = args.has_option("show_z0");
    bool cull_backfaces = args.has_option("cull_backfaces");
    int mesh_to_show = -1;

    // Create Window for rendering
    pangolin::CreateWindowAndBind("Main",w, h);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(w, h, f, f, w/2.0, h/2.0, 0.2, 1000),
        pangolin::ModelViewLookAt(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, pangolin::AxisY)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
            .SetHandler(&handler);

    // Load Geometry asynchronously
    std::vector<std::future<pangolin::Geometry>> geom_to_load;
    for(const auto& f : ExpandGlobOption(args["model"]))
    {
        geom_to_load.emplace_back( std::async(std::launch::async,[f](){
            return pangolin::LoadGeometry(f);
        }) );
    }

    // Render tree for holding object position
    RenderNode root;
    std::vector<std::shared_ptr<GlGeomRenderable>> renderables;
    pangolin::AxisDirection spin_other = pangolin::AxisNone;
    auto spin_transform = std::make_shared<SpinTransform>(spin_direction);
    auto show_renderable = [&](int index){
        mesh_to_show = index;
        for(size_t i=0; i < renderables.size(); ++i) {
            if(renderables[i]) renderables[i]->show = (index == -1 || index == (int)i) ? true : false;
        }
    };

    // Pull one piece of loaded geometry onto the GPU if ready
    Eigen::AlignedBox3f total_aabb;
    auto LoadGeometryToGpu = [&]()
    {
        for(auto& future_geom : geom_to_load) {
            if( future_geom.valid() && is_ready(future_geom) ) {
                auto geom = future_geom.get();
                auto aabb = pangolin::GetAxisAlignedBox(geom);
                total_aabb.extend(aabb);
                const Eigen::Vector3f center = total_aabb.center();
                const Eigen::Vector3f view = center + Eigen::Vector3f(1.2f,1.2f,1.2f) * std::max( (total_aabb.max() - center).norm(), (center - total_aabb.min()).norm());
                const auto mvm = pangolin::ModelViewLookAt(view[0], view[1], view[2], center[0], center[1], center[2], pangolin::AxisY);
                s_cam.SetModelViewMatrix(mvm);
                auto renderable = std::make_shared<GlGeomRenderable>(pangolin::ToGlGeometry(geom), aabb);
                renderables.push_back(renderable);
                RenderNode::Edge edge = { spin_transform, { renderable, {} } };
                root.edges.emplace_back(std::move(edge));
                break;
            }
        }
    };

    // Load Any matcap materials
    size_t matcap_index = 0;
    std::vector<pangolin::GlTexture> matcaps = TryLoad<pangolin::GlTexture>(ExpandGlobOption(args["matcap"]), [](const std::string& f){
        return pangolin::GlTexture(pangolin::LoadImage(f));
    });

    // Load Any Environment maps
    size_t envmap_index = 0;
    std::vector<pangolin::GlTexture> envmaps = TryLoad<pangolin::GlTexture>(ExpandGlobOption(args["envmap"]), [](const std::string& f){
        return pangolin::GlTexture(pangolin::LoadImage(f));
    });

    // GlSl Graphics shader program for display
    pangolin::GlSlProgram default_prog;
    auto LoadProgram = [&](const RenderMode mode){
        current_mode = mode;
        default_prog.ClearShaders();
        std::map<std::string,std::string> prog_defines;
        for(int i=0; i < (int)RenderMode::num_modes-1; ++i) {
            prog_defines[mode_names[i]] = std::to_string((int)mode == i);
        }
        default_prog.AddShader(pangolin::GlSlAnnotatedShader, default_model_shader, prog_defines);
        default_prog.Link();
    };
    LoadProgram(current_mode);

    pangolin::GlSlProgram env_prog;
    if(envmaps.size()) {
        env_prog.AddShader(pangolin::GlSlAnnotatedShader, equi_env_shader);
        env_prog.Link();
    }

    // Setup keyboard shortcuts.
    for(int i=0; i < (int)RenderMode::num_modes; ++i)
        pangolin::RegisterKeyPressCallback(mode_key[i], [&,i](){LoadProgram((RenderMode)i);});

    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + PANGO_KEY_RIGHT, [&](){if(renderables.size()) show_renderable((mesh_to_show + 1) % renderables.size());});
    pangolin::RegisterKeyPressCallback(PANGO_SPECIAL + PANGO_KEY_LEFT, [&](){if(renderables.size()) show_renderable((mesh_to_show + renderables.size()-1) % renderables.size());});
    pangolin::RegisterKeyPressCallback(' ', [&](){show_renderable(-1);});

    if(matcaps.size()) {
        pangolin::RegisterKeyPressCallback('=', [&](){matcap_index = (matcap_index+1)%matcaps.size();});
        pangolin::RegisterKeyPressCallback('-', [&](){matcap_index = (matcap_index+matcaps.size()-1)%matcaps.size();});
    }
    if(envmaps.size()) {
        pangolin::RegisterKeyPressCallback(']', [&](){envmap_index = ((envmap_index + 1) % envmaps.size());});
        pangolin::RegisterKeyPressCallback('[', [&](){envmap_index = ((envmap_index + envmaps.size()-1) % envmaps.size());});
    }
    pangolin::RegisterKeyPressCallback('s', [&](){std::swap(spin_transform->dir, spin_other);});
    pangolin::RegisterKeyPressCallback('b', [&](){show_bounds = !show_bounds;});
    pangolin::RegisterKeyPressCallback('0', [&](){cull_backfaces = !cull_backfaces;});

    // Show axis and axis planes
    pangolin::RegisterKeyPressCallback('a', [&](){show_axis = !show_axis;});
    pangolin::RegisterKeyPressCallback('x', [&](){show_x0 = !show_x0;});
    pangolin::RegisterKeyPressCallback('y', [&](){show_y0 = !show_y0;});
    pangolin::RegisterKeyPressCallback('z', [&](){show_z0 = !show_z0;});

    Eigen::Vector3d Pick_w = handler.Selected_P_w();
    std::vector<Eigen::Vector3d> Picks_w;

    while( !pangolin::ShouldQuit() )
    {
        if( (handler.Selected_P_w() - Pick_w).norm() > 1E-6)
        {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                      << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Load any pending geometry to the GPU.
        LoadGeometryToGpu();


        if(d_cam.IsShown()) {
            d_cam.Activate();

            if(cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }

            if(env_prog.Valid()) {
                glDisable(GL_DEPTH_TEST);
                env_prog.Bind();
                const Eigen::Matrix4d mvmat = s_cam.GetModelViewMatrix();
                const Eigen::Matrix3f R_env_cam = mvmat.block<3,3>(0,0).cast<float>().transpose();
                Eigen::Matrix3f Kinv;
                Kinv << 1.0/f, 0.0, -(w/2.0)/f,
                        0.0, 1.0/f, -(h/2.0)/f,
                        0.0, 0.0, 1.0;

                env_prog.SetUniform("R_env_camKinv", (R_env_cam*Kinv).eval());

                const GLint vertex_handle = env_prog.GetAttributeHandle("vertex");
                const GLint xy_handle = env_prog.GetAttributeHandle("xy");

                if(vertex_handle >= 0 && xy_handle >= 0 ) {
                    glActiveTexture(GL_TEXTURE0);
                    envmaps[envmap_index].Bind();
                    const GLfloat ndc[] = { 1.0f,1.0f,  -1.0f,1.0f,  -1.0f,-1.0f,  1.0f,-1.0f };
                    const GLfloat pix[] = { 0.0f,0.0f,  w,0.0f,  w,h,  0.0f,h };

                    glEnableVertexAttribArray(vertex_handle);
                    glVertexAttribPointer(vertex_handle, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), ndc );
                    glEnableVertexAttribArray(xy_handle);
                    glVertexAttribPointer(xy_handle, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), pix );
                    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
                    glDisableVertexAttribArray(vertex_handle);
                    glDisableVertexAttribArray(xy_handle);

                    env_prog.Unbind();
                    glEnable(GL_DEPTH_TEST);
                }
            }

            default_prog.Bind();
            render_tree(
                default_prog, root, s_cam.GetProjectionMatrix(), s_cam.GetModelViewMatrix(),
                matcaps.size() ? &matcaps[matcap_index] : nullptr
            );
            default_prog.Unbind();

            s_cam.Apply();
            if(show_x0) pangolin::glDraw_x0(10.0, 10);
            if(show_y0) pangolin::glDraw_y0(10.0, 10);
            if(show_z0) pangolin::glDraw_z0(10.0, 10);
            if(show_axis) pangolin::glDrawAxis(10.0);

            glDisable(GL_CULL_FACE);
        }

        pangolin::FinishFrame();
    }

    return 0;
}
