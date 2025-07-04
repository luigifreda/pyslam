#include <pangolin/display/image_view.h>
#include <pangolin/image/image_utils.h>
#include <pangolin/image/image_convert.h>

namespace pangolin
{

ImageView::ImageView()
    : offset_scale(0.0f, 1.0f), lastPressed(false), mouseReleased(false), mousePressed(false), overlayRender(true)
{
    SetHandler(this);
}

ImageView::~ImageView()
{
}

void ImageView::Render()
{
    LoadPending();

    glPushAttrib(GL_DEPTH_BITS);
    glDisable(GL_DEPTH_TEST);

    Activate();
    this->UpdateView();
    this->glSetViewOrtho();

    if(tex.IsValid())
    {
        if(offset_scale.first != 0.0 || offset_scale.second != 1.0)
        {
            pangolin::GlSlUtilities::OffsetAndScale(offset_scale.first, offset_scale.second);
        }
        else
        {
            glColor4f(1, 1, 1, 1);
        }

        this->glRenderTexture(tex);
        pangolin::GlSlUtilities::UseNone();
    }

    if(overlayRender)
    {
        this->glRenderOverlay();
    }

    if(extern_draw_function)
    {
        extern_draw_function(*this);
    }

    glPopAttrib();
}

void ImageView::Mouse(View& view, pangolin::MouseButton button, int x, int y, bool pressed, int button_state)
{
    ImageViewHandler::Mouse(view, button, x, y, pressed, button_state);

    mouseReleased = (!pressed && lastPressed);

    mousePressed = lastPressed = (pressed && button == pangolin::MouseButtonLeft);
}

void ImageView::Keyboard(View& view, unsigned char key, int x, int y, bool pressed)
{
    if(key == 'a')
    {
        if(!tex.IsValid())
        {
            std::cerr << "ImageViewHandler does not contain valid texture." << std::endl;
            return;
        }

        // compute scale
        const bool have_selection = std::isfinite(GetSelection().Area()) && std::abs(GetSelection().Area()) >= 4;
        const pangolin::XYRangef froi = have_selection ? GetSelection() : GetViewToRender();

        // Download texture so that we can take min / max
        pangolin::TypedImage img;
        tex.Download(img);
        offset_scale = pangolin::GetOffsetScale(img, pangolin::Round(froi), img.fmt);
    }
    else if(key == 'b')
    {
        if(!tex.IsValid())
        {
            std::cerr << "ImageViewHandler does not contain valid texture." << std::endl;
            return;
        }

        // compute scale
        const bool have_selection = std::isfinite(GetSelection().Area()) && std::abs(GetSelection().Area()) >= 4;
        const pangolin::XYRangef froi = have_selection ? GetSelection() : GetViewToRender();

        // Download texture so that we can take min / max
        pangolin::TypedImage img;
        tex.Download(img);
        std::pair<float, float> mm = pangolin::GetMinMax(img, pangolin::Round(froi), img.fmt);

        printf("Min / Max in Region: %f / %f\n", mm.first, mm.second);
    }
    else
    {
        pangolin::ImageViewHandler::Keyboard(view, key, x, y, pressed);
    }
}

pangolin::GlTexture& ImageView::Tex() {
    return tex;
}

ImageView& ImageView::SetImage(void* ptr, size_t w, size_t h, size_t pitch, pangolin::GlPixFormat img_fmt, bool delayed_upload )
{
    const size_t pix_bytes =
            pangolin::GlFormatChannels(img_fmt.glformat) * pangolin::GlDataTypeBytes(img_fmt.gltype);

    const bool convert_first = (img_fmt.gltype == GL_DOUBLE);

    if(delayed_upload || !pangolin::GetBoundWindow() || IsDevicePtr(ptr) || convert_first )
    {
        texlock.lock();
        if(!convert_first) {
            img_to_load = ManagedImage<unsigned char>(w,h,w*pix_bytes);
            PitchedCopy((char*)img_to_load.ptr, img_to_load.pitch, (char*)ptr, pitch, w * pix_bytes, h);
            img_fmt_to_load = img_fmt;
        }else if(img_fmt.gltype == GL_DOUBLE) {
            Image<double> double_image( (double*)ptr, w, h, pitch);
            img_to_load.OwnAndReinterpret(ImageConvert<float>(double_image));
            img_fmt_to_load = GlPixFormat::FromType<float>();
        }else{
            pango_print_warn("TextureView: Unable to display image.\n");
        }
        texlock.unlock();
        return *this;
    }

    PANGO_ASSERT(pitch % pix_bytes == 0);
    const size_t stride = pitch / pix_bytes;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, stride);

    // Initialise if it didn't already exist or the size was too small
    if(!tex.tid || tex.width != (int)w || tex.height != (int)h ||
       tex.internal_format != img_fmt.scalable_internal_format)
    {
        fmt = img_fmt;
        SetDimensions(w, h);
        SetAspect((float)w / (float)h);
        tex.Reinitialise(w, h, img_fmt.scalable_internal_format, true, 0, img_fmt.glformat, img_fmt.gltype, ptr);
    }
    else
    {
        tex.Upload(ptr, img_fmt.glformat, img_fmt.gltype);
    }

    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    return *this;
}

ImageView& ImageView::SetImage(const pangolin::Image<unsigned char>& img, const pangolin::GlPixFormat& glfmt, bool delayed_upload )
{
    return SetImage(img.ptr, img.w, img.h, img.pitch, glfmt, delayed_upload);
}

ImageView& ImageView::SetImage(const pangolin::TypedImage& img, bool delayed_upload )
{
    return SetImage(img.ptr, img.w, img.h, img.pitch, pangolin::GlPixFormat(img.fmt), delayed_upload);
}

ImageView& ImageView::SetImage(const pangolin::GlTexture& texture)
{
    // Initialise if it didn't already exist or the size was too small
    if(!tex.tid || tex.width != texture.width || tex.height != texture.height ||
       tex.internal_format != texture.internal_format)
    {
        SetDimensions(texture.width, texture.height);
        SetAspect((float)texture.width / (float)texture.height);
        tex.Reinitialise(texture.width, texture.height, texture.internal_format, true);
    }

    glCopyImageSubData(
            texture.tid, GL_TEXTURE_2D, 0, 0, 0, 0, tex.tid, GL_TEXTURE_2D, 0, 0, 0, 0, tex.width, tex.height, 1);

    return *this;
}

void ImageView::LoadPending()
{
    if(img_to_load.ptr)
    {
        // Scoped lock
        texlock.lock();
        SetImage(img_to_load, img_fmt_to_load, false);
        img_to_load.Deallocate();
        texlock.unlock();
    }
}

ImageView& ImageView::Clear()
{
    tex.Delete();
    return *this;
}

std::pair<float, float>& ImageView::GetOffsetScale() {
    return offset_scale;
}

bool ImageView::MouseReleased() const {
    return mouseReleased;
}

bool ImageView::MousePressed() const {
    return mousePressed;
}

void ImageView::SetRenderOverlay(const bool& val) {
    overlayRender = val;
}

}
