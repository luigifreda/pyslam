#pragma once

#include <pangolin/display/display.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/gl/glformattraits.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/image/image_utils.h>

#include <mutex>

namespace pangolin
{

class ImageView : public pangolin::View, public pangolin::ImageViewHandler
{
  public:
    ImageView();

    ~ImageView();

    void Render() override;

    void Mouse(View& view, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) override;

    void Keyboard(View& view, unsigned char key, int x, int y, bool pressed) override;

    pangolin::GlTexture& Tex();

    ImageView& SetImage(void* ptr, size_t w, size_t h, size_t pitch, pangolin::GlPixFormat img_fmt, bool delayed_upload = false);

    ImageView& SetImage(const pangolin::Image<unsigned char>& img, const pangolin::GlPixFormat& glfmt, bool delayed_upload = false);

    template<typename T> inline
    ImageView& SetImage(const pangolin::Image<T>& img, bool delayed_upload = false)
    {
        return SetImage(img.template UnsafeReinterpret<unsigned char>(), GlPixFormat::FromType<T>(), delayed_upload);
    }

    ImageView& SetImage(const pangolin::TypedImage& img, bool delayed_upload = false);

    ImageView& SetImage(const pangolin::GlTexture& texture);

    void LoadPending();

    ImageView& Clear();

    std::pair<float, float>& GetOffsetScale();

    bool MouseReleased() const;

    bool MousePressed() const;

    void SetRenderOverlay(const bool& val);

//  private:
    // img_to_load contains image data that should be uploaded to the texture on
    // the next render cycle. The data is owned by this object and should be
    // freed after use.
    pangolin::ManagedImage<unsigned char> img_to_load;
    pangolin::GlPixFormat img_fmt_to_load;

    std::pair<float, float> offset_scale;
    pangolin::GlPixFormat fmt;
    pangolin::GlTexture tex;
    bool lastPressed;
    bool mouseReleased;
    bool mousePressed;
    bool overlayRender;

    std::mutex texlock;
};

}
