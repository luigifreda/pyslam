#include <pangolin/platform.h>
#include <pangolin/display/display.h>
#include <pangolin/display/display_internal.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/gl/colour.h>
#include <pangolin/gl/gldraw.h>

#include <wayland-client.h>
#include <wayland-egl.h>
#include <EGL/egl.h>
#include <xkbcommon/xkbcommon.h>
#include <wayland-cursor.h>
#include <linux/input.h>
#include <sys/mman.h>

#include <mutex>
#include <string.h>
#include <unistd.h>
#include <cstdlib>

#if USE_WL_XDG
#include <xdg-shell-client-protocol.h>
#endif

#define WAYLAND_VERSION_GE(MAJ, MIN) WAYLAND_VERSION_MAJOR >= MAJ && WAYLAND_VERSION_MINOR >= MIN

namespace pangolin {

extern __thread PangolinGl* context;

namespace wayland {

static const std::map<enum wl_shell_surface_resize, std::string> resize_cursor = {
    {WL_SHELL_SURFACE_RESIZE_NONE, "grabbing"},
    {WL_SHELL_SURFACE_RESIZE_TOP, "top_side"},
    {WL_SHELL_SURFACE_RESIZE_BOTTOM, "bottom_side"},
    {WL_SHELL_SURFACE_RESIZE_LEFT, "left_side"},
    {WL_SHELL_SURFACE_RESIZE_TOP_LEFT, "top_left_corner"},
    {WL_SHELL_SURFACE_RESIZE_BOTTOM_LEFT, "bottom_left_corner"},
    {WL_SHELL_SURFACE_RESIZE_RIGHT, "right_side"},
    {WL_SHELL_SURFACE_RESIZE_TOP_RIGHT, "top_right_corner"},
    {WL_SHELL_SURFACE_RESIZE_BOTTOM_RIGHT, "bottom_right_corner"}
};

struct ButtonSurface {
    struct wl_surface *surface;
    struct wl_subsurface *subsurface;
    struct wl_egl_window *egl_window;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;
    const int32_t x, y;
    const uint width, height;
    pangolin::Colour colour;

    enum type {
        CLOSE = 100,
        MAXIMISE
    } function;

    ButtonSurface(wl_compositor* compositor, wl_subcompositor* subcompositor,
                  wl_surface* source, EGLDisplay egl_display, EGLConfig config,
                  int32_t x, int32_t y, uint width, uint height,
                  type fnct, pangolin::Colour colour
                  ) :
        egl_display(egl_display),
        x(x), y(y),
        width(width), height(height),
        colour(colour),
        function(fnct)
    {
        surface = wl_compositor_create_surface(compositor);
        subsurface = wl_subcompositor_get_subsurface(subcompositor, surface, source);
        wl_subsurface_set_desync(subsurface);
        egl_context = eglCreateContext (egl_display, config, EGL_NO_CONTEXT, nullptr);
        egl_window = wl_egl_window_create(surface, width, height);
        egl_surface = eglCreateWindowSurface(egl_display, config, (EGLNativeWindowType)egl_window, nullptr);
    }

    ~ButtonSurface() {
        if(egl_surface) eglDestroySurface(egl_display, egl_surface);
        if(egl_window)  wl_egl_window_destroy(egl_window);
        if(egl_context) eglDestroyContext(egl_display, egl_context);

        if(subsurface)  wl_subsurface_destroy(subsurface);
        if(surface)     wl_surface_destroy(surface);
    }

    void reposition(const int main_w) const {
        wl_subsurface_set_position(subsurface, main_w-x-width, y);
    }

    void draw() const {
        eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
        glClearColor(colour.r, colour.g, colour.b, colour.a);
        glClear(GL_COLOR_BUFFER_BIT);
        switch (function) {
        case CLOSE:
            glLineWidth(3);
            glColor3f(1, 1, 1);
            glBegin(GL_LINES);
            glVertex2f(-1, -1);
            glVertex2f(1, 1);
            glVertex2f(1, -1);
            glVertex2f(-1, 1);
            glEnd();
            break;
        case MAXIMISE:
            glLineWidth(2);
            glColor3f(0, 0, 0);
            glBegin(GL_LINE_LOOP);
            glVertex2f(-0.7f, -0.7f);
            glVertex2f(0.7f, -0.7f);
            glVertex2f(0.7f, 0.7f);
            glVertex2f(-0.7f, 0.7f);
            glEnd();
            glLineWidth(3);
            glBegin(GL_LINES);
            glVertex2f(+0.7f, +0.7f);
            glVertex2f(-0.7f, +0.7f);
            glEnd();
            break;
        }
        eglSwapBuffers(egl_display, egl_surface);
    }
};

struct DecorationSurface {
    struct wl_surface *surface;
    struct wl_subsurface *subsurface;
    struct wl_egl_window *egl_window;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;
    pangolin::Colour colour;
    uint border_size;
    uint title_bar_size;

    enum wl_shell_surface_resize function;

    DecorationSurface(wl_compositor* compositor, wl_subcompositor* subcompositor,
                      wl_surface* source, EGLDisplay egl_display, EGLConfig config,
                      const uint _border_size, const uint _title_bar_size,
                      enum wl_shell_surface_resize type, pangolin::Colour _colour
                      ) :
        egl_display(egl_display),
        colour(_colour),
        border_size(_border_size),
        title_bar_size(_title_bar_size),
        function(type)
    {
        surface = wl_compositor_create_surface(compositor);
        subsurface = wl_subcompositor_get_subsurface(subcompositor, surface, source);
        wl_subsurface_set_desync(subsurface);
        egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, nullptr);
        egl_window = wl_egl_window_create(surface, 50, 50);
        egl_surface = eglCreateWindowSurface(egl_display, config, (EGLNativeWindowType)egl_window, nullptr);
    }

    ~DecorationSurface() {
        if(egl_surface) eglDestroySurface(egl_display, egl_surface);
        if(egl_window)  wl_egl_window_destroy(egl_window);
        if(egl_context) eglDestroyContext(egl_display, egl_context);

        if(subsurface)  wl_subsurface_destroy(subsurface);
        if(surface)     wl_surface_destroy(surface);
    }

    void calc_dim(const int main_w, const int main_h, int &x, int &y, int &w, int &h) const {
        // get position and dimension from type and main surface
        switch (function) {
        case WL_SHELL_SURFACE_RESIZE_NONE:
            x=0; y=-title_bar_size;
            w=main_w; h=title_bar_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_TOP:
            x=0; y=-title_bar_size-border_size;
            w=main_w; h=border_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_BOTTOM:
            x=0; y=main_h;
            w=main_w; h=border_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_LEFT:
            x=-border_size; y=-title_bar_size;
            w=border_size; h=main_h+title_bar_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_TOP_LEFT:
            x=-border_size; y=-border_size-title_bar_size;
            w=border_size; h=border_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_BOTTOM_LEFT:
            x=-border_size; y=main_h;
            w=border_size; h=border_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_RIGHT:
            x=main_w; y=-title_bar_size;
            w=border_size; h=main_h+title_bar_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_TOP_RIGHT:
            x=main_w; y=-border_size-title_bar_size;
            w=border_size; h=border_size;
            break;
        case WL_SHELL_SURFACE_RESIZE_BOTTOM_RIGHT:
            x=main_w; y=main_h;
            w=border_size; h=border_size;
            break;
        }
    }

    void resize(const int main_w, const int main_h) const {
        int x=0, y=0, w=0, h=0;
        calc_dim(main_w, main_h, x, y, w, h);
        wl_subsurface_set_position(subsurface, x, y);
        wl_egl_window_resize(egl_window, w, h, 0, 0);
    }

    void draw() const {
        eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
        glClearColor(colour.r, colour.g, colour.b, colour.a);
        glClear(GL_COLOR_BUFFER_BIT);
        eglSwapBuffers(egl_display, egl_surface);
    }
};

struct Decoration {
    Decoration(const uint border_size,
               const uint title_size,
               const pangolin::Colour colour,
               wl_compositor* compositor,
               wl_subcompositor* subcompositor,
               wl_surface* surface,
               EGLDisplay egl_display,
               EGLConfig config
               ) :
        border_size(border_size),
        title_size(title_size),
        colour(colour),
        egl_display(egl_display),
        compositor(compositor),
        subcompositor(subcompositor),
        surface(surface),
        config(config)
    { }

    void create() {
        // reserve memory to prevent that DecorationSurface's destructor gets
        // called by 'emplace_back'
        decorations.reserve(9);

        // title bar, 2D movement
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_NONE, colour);

        // sides, 1D resizing
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_LEFT, colour);
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_RIGHT, colour);
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_TOP, colour);
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_BOTTOM, colour);

        // corners, 2D resizing
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_TOP_LEFT, colour);
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_TOP_RIGHT, colour);
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_BOTTOM_LEFT, colour);
        decorations.emplace_back(compositor, subcompositor, surface, egl_display, config, border_size, title_size, WL_SHELL_SURFACE_RESIZE_BOTTOM_RIGHT, colour);

        // buttons
        buttons.reserve(2);
        buttons.emplace_back(compositor, subcompositor, decorations[0].surface, egl_display, config, 5, 1, button_width, button_height,  ButtonSurface::type::CLOSE, pangolin::Colour(0, 110/255.0, 182/255.0));
        buttons.emplace_back(compositor, subcompositor, decorations[0].surface, egl_display, config, 10+button_width, 1, button_width, button_height,  ButtonSurface::type::MAXIMISE, pangolin::Colour(1.0, 204/255.0f, 0));
    }

    void destroy() {
        decorations.clear();
        buttons.clear();
    }

    void resize(const int32_t width, const int32_t height) {
        for(const DecorationSurface &d : decorations) { d.resize(width, height); }
        for(const ButtonSurface &b : buttons) { b.reposition(width); }
    }

    void draw() {
        for(const DecorationSurface &d : decorations) { d.draw(); }
        for(const ButtonSurface &b : buttons) { b.draw(); }
    }

    void setTypeFromSurface(const wl_surface *surface) {
        for(const DecorationSurface &d : decorations) {
            if(d.surface==surface) {
                last_type = d.function;
                return;
            }
        }
        for(const ButtonSurface &b : buttons) {
            if(b.surface==surface) {
                last_type = b.function;
                return;
            }
        }
        // surface is not part of the window decoration
        last_type = -1;
    }

    const std::string getCursorForCurrentSurface() const {
        return resize_cursor.count((enum wl_shell_surface_resize)last_type) ? resize_cursor.at((enum wl_shell_surface_resize)last_type) : "left_ptr";
    }

    std::vector<DecorationSurface> decorations;
    int last_type;

    std::vector<ButtonSurface> buttons;

    const uint border_size;
    const uint title_size;
    const pangolin::Colour colour;
    EGLDisplay egl_display;
    wl_compositor* compositor;
    wl_subcompositor* subcompositor;
    wl_surface* surface;
    EGLConfig config;

    static const uint button_width;
    static const uint button_height;
};

const uint Decoration::button_width = 25;
const uint Decoration::button_height = 15;

struct WaylandDisplay {
    WaylandDisplay(const int width, const int height, const std::string title = "");

    ~WaylandDisplay();

    struct wl_display *wdisplay = nullptr;
    struct wl_registry *wregistry = nullptr;
    struct wl_compositor *wcompositor = nullptr;
    struct wl_subcompositor *wsubcompositor = nullptr;
    struct wl_surface *wsurface = nullptr;
    struct wl_egl_window *egl_window = nullptr;
#if USE_WL_XDG
    struct xdg_wm_base *xshell = nullptr;
    struct xdg_surface *xshell_surface = nullptr;
    struct xdg_toplevel *xshell_toplevel = nullptr;
#else
    struct wl_shell *wshell = nullptr;
    struct wl_shell_surface *wshell_surface = nullptr;
#endif

    struct wl_seat *wseat = nullptr;
    struct wl_keyboard *wkeyboard = nullptr;
    struct wl_pointer *pointer = nullptr;

    // for cursor
    struct wl_shm *shm = nullptr;
    struct wl_cursor_theme *cursor_theme = nullptr;
    struct wl_surface *cursor_surface = nullptr;

    // xkbcommon
    struct xkb_context *xkb_context = nullptr;
    struct xkb_keymap *keymap = nullptr;
    struct xkb_state *xkb_state = nullptr;

    std::unique_ptr<Decoration> decoration;

    bool pressed = false;
    int lastx=0;
    int lasty=0;

    std::vector<EGLConfig> egl_configs;
    EGLSurface egl_surface = nullptr;
    EGLContext egl_context = nullptr;
    EGLDisplay egl_display = nullptr;

    EGLint width, height;
    bool is_fullscreen;
    bool is_maximised;
    static constexpr EGLint attribs[] = {
        EGL_RENDERABLE_TYPE , EGL_OPENGL_BIT,
        EGL_RED_SIZE        , 8,
        EGL_GREEN_SIZE      , 8,
        EGL_BLUE_SIZE       , 8,
        EGL_ALPHA_SIZE      , 8,
        EGL_DEPTH_SIZE      , 24,
        EGL_STENCIL_SIZE    , 8,
        EGL_NONE
    };
};

constexpr EGLint WaylandDisplay::attribs[];

struct WaylandWindow : public PangolinGl
{
    WaylandWindow(const int width, const int height, std::unique_ptr<WaylandDisplay> display);

    ~WaylandWindow() override;

    void ToggleFullscreen() override;

    void Move(const int x, const int y) override;

    void Resize(const unsigned int w, const unsigned int h) override;

    void MakeCurrent() override;

    void RemoveCurrent() override;

    void SwapBuffers() override;

    void ProcessEvents() override;

    std::unique_ptr<WaylandDisplay> display;
};

// map wayland ids to pangolin ids
static const std::map<uint,int> wl_button_ids = {
    {BTN_LEFT, 0},
    {BTN_MIDDLE, 1},
    {BTN_RIGHT, 2},
};

static const std::map<uint,KeyModifier> wl_key_mod_ids = {
    {KEY_LEFTSHIFT, KeyModifierShift},
    {KEY_RIGHTSHIFT, KeyModifierShift},
    {KEY_LEFTCTRL, KeyModifierCtrl},
    {KEY_RIGHTCTRL, KeyModifierCtrl},
    {KEY_LEFTALT, KeyModifierAlt},
    {KEY_RIGHTALT, KeyModifierAlt},
};

static const std::map<uint,int> wl_key_special_ids = {
    {KEY_F1, PANGO_KEY_F1},
    {KEY_F2, PANGO_KEY_F2},
    {KEY_F3, PANGO_KEY_F3},
    {KEY_F4, PANGO_KEY_F4},
    {KEY_F5, PANGO_KEY_F5},
    {KEY_F6, PANGO_KEY_F6},
    {KEY_F7, PANGO_KEY_F7},
    {KEY_F8, PANGO_KEY_F8},
    {KEY_F9, PANGO_KEY_F9},
    {KEY_F10, PANGO_KEY_F10},
    {KEY_F11, PANGO_KEY_F11},
    {KEY_F12, PANGO_KEY_F12},

    {KEY_LEFT, PANGO_KEY_LEFT},
    {KEY_UP, PANGO_KEY_UP},
    {KEY_RIGHT, PANGO_KEY_RIGHT},
    {KEY_DOWN, PANGO_KEY_DOWN},

    {KEY_PAGEUP, PANGO_KEY_PAGE_UP},
    {KEY_PAGEDOWN, PANGO_KEY_PAGE_DOWN},
    {KEY_HOME, PANGO_KEY_HOME},
    {KEY_END, PANGO_KEY_END},
    {KEY_INSERT, PANGO_KEY_INSERT},
};

#if USE_WL_XDG
static void handle_configure_toplevel(void *data, struct xdg_toplevel */*xdg_toplevel*/, int32_t width, int32_t height, struct wl_array */*states*/) {
#else
static void handle_configure(void *data, struct wl_shell_surface */*shell_surface*/, uint32_t /*edges*/, int32_t width, int32_t height) {
#endif

    const static uint min_width = 70;
    const static uint min_height = 70;

    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    const bool provided = !(width==0 && height==0);

    int main_w, main_h;
    if(provided) {
        // use the provided dimensions
        if(w->is_fullscreen) {
            // without decorations
            main_w = width;
            main_h = height;
        }
        else {
            // with decorations
            main_w = std::max(width-int(2*w->decoration->border_size), int(min_width));
            main_h = std::max(height-2*int(w->decoration->border_size)-int(w->decoration->title_size), int(min_height));
        }
    }
    else {
        // restore from saved dimensions
        main_w = std::max((w->width)-int(2*w->decoration->border_size), int(min_width));
        main_h = std::max((w->height)-2*int(w->decoration->border_size)-int(w->decoration->title_size), int(min_height));
    }

    // resize main surface
    wl_egl_window_resize(w->egl_window, main_w, main_h, 0, 0);

    // set opaque region
    struct wl_region* wregion = wl_compositor_create_region(w->wcompositor);
    wl_region_add(wregion, 0, 0, main_w, main_h);
    wl_surface_set_opaque_region(w->wsurface, wregion);
    wl_region_destroy(wregion);

    // resize all decoration elements
    w->decoration->resize(main_w, main_h);

    // notify Panglin views about resized area
    pangolin::process::Resize(main_w, main_h);
}

#if USE_WL_XDG
static void handle_configure(void */*data*/, struct xdg_surface *xdg_surface, uint32_t serial) {
    xdg_surface_ack_configure(xdg_surface, serial);
}

static const struct xdg_surface_listener shell_surface_listener = {
    handle_configure,
};

static void handle_toplevel_close(void */*data*/, struct xdg_toplevel */*xdg_toplevel*/) {
    pangolin::QuitAll();
}

static const struct xdg_toplevel_listener toplevel_listener = {
    .configure = handle_configure_toplevel,
    .close = handle_toplevel_close,
};

static void xdg_wm_base_ping(void */*data*/, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
    xdg_wm_base_pong(xdg_wm_base, serial);
}

static const struct xdg_wm_base_listener shell_listener = {
    .ping = xdg_wm_base_ping
};
#else
static void handle_ping(void */*data*/, struct wl_shell_surface *shell_surface, uint32_t serial) {
    wl_shell_surface_pong(shell_surface, serial);
}

static void handle_popup_done(void */*data*/, struct wl_shell_surface */*shell_surface*/) { }

static const struct wl_shell_surface_listener shell_surface_listener = {
    handle_ping,
    handle_configure,
    handle_popup_done
};
#endif

static void pointer_handle_enter(void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t /*sx*/, wl_fixed_t /*sy*/) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);
    w->decoration->setTypeFromSurface(surface);

    const std::string cursor = w->decoration->getCursorForCurrentSurface();

    const auto image = wl_cursor_theme_get_cursor(w->cursor_theme, cursor.c_str())->images[0];
    wl_pointer_set_cursor(pointer, serial, w->cursor_surface, image->hotspot_x, image->hotspot_y);
    wl_surface_attach(w->cursor_surface, wl_cursor_image_get_buffer(image), 0, 0);
    wl_surface_damage(w->cursor_surface, 0, 0, image->width, image->height);
    wl_surface_commit(w->cursor_surface);
}

static void pointer_handle_leave(void *data, struct wl_pointer */*pointer*/, uint32_t /*serial*/, struct wl_surface */*surface*/) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);
    w->pressed = false;
}

static void pointer_handle_motion(void *data, struct wl_pointer */*pointer*/, uint32_t /*time*/, wl_fixed_t sx, wl_fixed_t sy) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    w->lastx=wl_fixed_to_int(sx);
    w->lasty=wl_fixed_to_int(sy);
    if(w->pressed) {
        pangolin::process::MouseMotion(w->lastx, w->lasty);
    }
    else {
        pangolin::process::PassiveMouseMotion(w->lastx, w->lasty);
    }
}

static void pointer_handle_button(void *data, struct wl_pointer */*wl_pointer*/, uint32_t serial, uint32_t /*time*/, uint32_t button, uint32_t state) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    if(w->decoration->last_type<0) {
        // input goes to pangoling view
        if(!wl_button_ids.count(button))
            return;

        w->pressed = (state==WL_POINTER_BUTTON_STATE_PRESSED);
        pangolin::process::Mouse(wl_button_ids.at(button), (state==WL_POINTER_BUTTON_STATE_RELEASED), w->lastx, w->lasty);
    }
    else {
        // input goes to window decoration
        // resizing using window decoration
        if((button==BTN_LEFT) && (state==WL_POINTER_BUTTON_STATE_PRESSED)) {
            switch (w->decoration->last_type) {
            case WL_SHELL_SURFACE_RESIZE_NONE:
#if USE_WL_XDG
                xdg_toplevel_move(w->xshell_toplevel, w->wseat, serial);
#else
                wl_shell_surface_move(w->wshell_surface, w->wseat, serial);
#endif
                break;
            case WL_SHELL_SURFACE_RESIZE_TOP:
            case WL_SHELL_SURFACE_RESIZE_BOTTOM:
            case WL_SHELL_SURFACE_RESIZE_LEFT:
            case WL_SHELL_SURFACE_RESIZE_TOP_LEFT:
            case WL_SHELL_SURFACE_RESIZE_BOTTOM_LEFT:
            case WL_SHELL_SURFACE_RESIZE_RIGHT:
            case WL_SHELL_SURFACE_RESIZE_TOP_RIGHT:
            case WL_SHELL_SURFACE_RESIZE_BOTTOM_RIGHT:
#if USE_WL_XDG
                xdg_toplevel_resize(w->xshell_toplevel, w->wseat, serial, w->decoration->last_type);
#else
                wl_shell_surface_resize(w->wshell_surface, w->wseat, serial, w->decoration->last_type);
#endif
                break;
            case ButtonSurface::type::CLOSE:
                pangolin::QuitAll();
                break;
            case ButtonSurface::type::MAXIMISE:
                w->is_maximised = !w->is_maximised;
                if(w->is_maximised) {
#if USE_WL_XDG
                    xdg_toplevel_set_maximized(w->xshell_toplevel);
#else
                    // store original window size
                    wl_egl_window_get_attached_size(w->egl_window, &w->width, &w->height);
                    wl_shell_surface_set_maximized(w->wshell_surface, nullptr);
#endif
                }
                else {
#if USE_WL_XDG
                    xdg_toplevel_unset_maximized(w->xshell_toplevel);
#else
                    wl_shell_surface_set_toplevel(w->wshell_surface);
                    handle_configure(data, nullptr, 0,
                                     w->width+2*w->decoration->border_size,
                                     w->height+2*w->decoration->border_size+w->decoration->title_size);
#endif
                }

                break;
            }
        }
    }
}

static void pointer_handle_axis(void *data, struct wl_pointer */*wl_pointer*/, uint32_t /*time*/, uint32_t axis, wl_fixed_t value) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    int button_id = -1;
    switch (axis) {
    case REL_X: button_id = (value<0) ? 3 : 4; break;   // up, down
    case REL_Y: button_id = (value<0) ? 5 : 6; break;   // left, right
    }

    if(button_id>0) {
        pangolin::process::Mouse(button_id, 0, w->lastx, w->lasty);
    }
}

#if WAYLAND_VERSION_GE(1,12)

static void pointer_handle_frame(void */*data*/, struct wl_pointer */*wl_pointer*/) { }

static void pointer_handle_axis_source(void */*data*/, struct wl_pointer */*wl_pointer*/, uint32_t /*axis_source*/) { }

static void pointer_handle_axis_stop(void */*data*/, struct wl_pointer */*wl_pointer*/, uint32_t /*time*/, uint32_t /*axis*/) { }

static void pointer_handle_axis_discrete(void */*data*/, struct wl_pointer */*wl_pointer*/, uint32_t /*axis*/, int32_t /*discrete*/) { }

#endif

static const struct wl_pointer_listener pointer_listener = {
    pointer_handle_enter,
    pointer_handle_leave,
    pointer_handle_motion,
    pointer_handle_button,
    pointer_handle_axis,
#if WAYLAND_VERSION_GE(1,12)
    pointer_handle_frame,
    pointer_handle_axis_source,
    pointer_handle_axis_stop,
    pointer_handle_axis_discrete,
#endif
};

static void keyboard_handle_keymap(void *data, struct wl_keyboard */*keyboard*/, uint32_t /*format*/, int fd, uint32_t size) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    char *keymap_string = static_cast<char*>(mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
    xkb_keymap_unref(w->keymap);
    w->keymap = xkb_keymap_new_from_string(w->xkb_context, keymap_string, XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
    munmap(keymap_string, size);
    close(fd);
    xkb_state_unref(w->xkb_state);
    w->xkb_state = xkb_state_new(w->keymap);
}

static void keyboard_handle_enter(void */*data*/, struct wl_keyboard */*keyboard*/, uint32_t /*serial*/, struct wl_surface */*surface*/, struct wl_array */*keys*/) { }

static void keyboard_handle_leave(void */*data*/, struct wl_keyboard */*keyboard*/, uint32_t /*serial*/, struct wl_surface */*surface*/) { }

static void keyboard_handle_key(void *data, struct wl_keyboard */*keyboard*/, uint32_t /*serial*/, uint32_t /*time*/, uint32_t key, uint32_t state) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    // modifier keys
    if(wl_key_mod_ids.count(key)) {
        if(state==WL_KEYBOARD_KEY_STATE_PRESSED) {
            pangolin::context->mouse_state |=  wl_key_mod_ids.at(key);
        }
        else if (state==WL_KEYBOARD_KEY_STATE_RELEASED) {
            pangolin::context->mouse_state &= ~wl_key_mod_ids.at(key);
        }
        return;
    }

    // character and special keys
    int pango_key = -1;
    if(wl_key_special_ids.count(key)) {
        // special keys
        pango_key = PANGO_SPECIAL + wl_key_special_ids.at(key);
    }
    else {
        // character keys
        const uint32_t utf32 = xkb_state_key_get_utf32(w->xkb_state, key+8);
        // filter non-ASCII
        if(utf32>0 && utf32<=127) {
            pango_key = int(utf32);
        }
    }

    if(pango_key>0) {
        if(state==WL_KEYBOARD_KEY_STATE_PRESSED) {
            pangolin::process::Keyboard(uint8_t(pango_key), w->lastx, w->lasty);
        }else if(state==WL_KEYBOARD_KEY_STATE_RELEASED){
            pangolin::process::KeyboardUp(uint8_t(pango_key), w->lastx, w->lasty);
        }
    }
}

static void keyboard_handle_modifiers(void *data, struct wl_keyboard */*keyboard*/, uint32_t /*serial*/, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    xkb_state_update_mask(w->xkb_state, mods_depressed, mods_latched, mods_locked, 0, 0, group);
}

#if WAYLAND_VERSION_GE(1,12)

static void keyboard_handle_repeat_info(void */*data*/, struct wl_keyboard */*wl_keyboard*/, int32_t /*rate*/, int32_t /*delay*/) { }

#endif

static const struct wl_keyboard_listener keyboard_listener = {
    keyboard_handle_keymap,
    keyboard_handle_enter,
    keyboard_handle_leave,
    keyboard_handle_key,
    keyboard_handle_modifiers,
#if WAYLAND_VERSION_GE(1,12)
    keyboard_handle_repeat_info,
#endif
};

static void seat_handle_capabilities(void *data, struct wl_seat *seat, uint32_t caps1) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    enum wl_seat_capability caps;
    caps = (enum wl_seat_capability)caps1;
    if (caps & WL_SEAT_CAPABILITY_KEYBOARD) {
        w->wkeyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(w->wkeyboard, &keyboard_listener, data);
    } else {
        wl_keyboard_destroy(w->wkeyboard);
        w->wkeyboard = nullptr;
    }
    if (caps & WL_SEAT_CAPABILITY_POINTER) {
        w->pointer = wl_seat_get_pointer(seat);
        w->cursor_surface = wl_compositor_create_surface(w->wcompositor);
        wl_pointer_add_listener(w->pointer, &pointer_listener, data);
    } else {
        wl_pointer_destroy(w->pointer);
        w->pointer = nullptr;
    }
}

static void seat_handle_name(void */*data*/, struct wl_seat */*wl_seat*/, const char */*name*/) { }

static const struct wl_seat_listener seat_listener = {
    seat_handle_capabilities,
    seat_handle_name,
};

static void global_registry_handler(void *data, struct wl_registry *registry, uint32_t id, const char *interface, uint32_t version) {
    WaylandDisplay* const w = static_cast<WaylandDisplay*>(data);

    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        w->wcompositor = reinterpret_cast<wl_compositor*> (wl_registry_bind(registry, id, &wl_compositor_interface, version));
    }
    else if (strcmp(interface, wl_subcompositor_interface.name) == 0) {
        w->wsubcompositor = static_cast<wl_subcompositor*>(wl_registry_bind(registry, id, &wl_subcompositor_interface, version));
    }
#if USE_WL_XDG
    else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        w->xshell = reinterpret_cast<xdg_wm_base*> (wl_registry_bind(registry, id, &xdg_wm_base_interface, version));
    }
#else
    else if (strcmp(interface, wl_shell_interface.name) == 0) {
        w->wshell = reinterpret_cast<wl_shell*> (wl_registry_bind(registry, id, &wl_shell_interface, version));
    }
#endif
    else if (strcmp(interface, wl_seat_interface.name) == 0) {
        w->wseat = reinterpret_cast<wl_seat*>(wl_registry_bind(registry, id, &wl_seat_interface, version));
        wl_seat_add_listener(w->wseat, &seat_listener, data);
    }
    else if (strcmp(interface, wl_shm_interface.name) == 0) {
        w->shm = static_cast<wl_shm*>(wl_registry_bind(registry, id, &wl_shm_interface, version));
        w->cursor_theme = wl_cursor_theme_load(nullptr, 16, w->shm);
    }
}

static void global_registry_remover(void */*data*/, struct wl_registry */*registry*/, uint32_t /*id*/) { }

static const struct wl_registry_listener wregistry_listener = {
    global_registry_handler,
    global_registry_remover
};

WaylandDisplay::WaylandDisplay(const int width, const int height, const std::string title) : width(width), height(height) {
    xkb_context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);

    wdisplay = wl_display_connect(nullptr);
    if (wdisplay == nullptr) {
        throw std::runtime_error("Cannot connect to Wayland compositor!");
    }

    wregistry = wl_display_get_registry(wdisplay);
    wl_registry_add_listener(wregistry, &wregistry_listener, this);

    wl_display_roundtrip(wdisplay);

    egl_display = eglGetDisplay((EGLNativeDisplayType)wdisplay);
    if(!egl_display) {
        std::cerr << "Failed to open EGL display" << std::endl;
    }

    EGLint major, minor;
    if(eglInitialize(egl_display, &major, &minor)==EGL_FALSE) {
        std::cerr << "EGL init failed" << std::endl;
    }

    if(eglBindAPI(EGL_OPENGL_API)==EGL_FALSE) {
        std::cerr << "EGL bind failed" << std::endl;
    }

    EGLint count;
    eglGetConfigs(egl_display, nullptr, 0, &count);

    egl_configs.resize(count);

    EGLint numConfigs;
    eglChooseConfig(egl_display, attribs, egl_configs.data(), count, &numConfigs);

    egl_context = eglCreateContext(egl_display, egl_configs[0], EGL_NO_CONTEXT, nullptr);

    wsurface = wl_compositor_create_surface(wcompositor);

    egl_window = wl_egl_window_create(wsurface, width, height);
    if(!egl_window) {
        std::cerr << "Cannot create EGL window" << std::endl;
    }

    egl_surface = eglCreateWindowSurface(egl_display, egl_configs[0], (EGLNativeWindowType)egl_window, nullptr);
    if (egl_surface == EGL_NO_SURFACE) {
        std::cerr << "Cannot create EGL surface" << std::endl;
    }

    if(
#if USE_WL_XDG
    xshell
#else
    wshell
#endif
    ==nullptr) {
        throw std::runtime_error("No Wayland shell available!");
    }

#if USE_WL_XDG
    xdg_wm_base_add_listener(xshell, &shell_listener, this);
    xshell_surface = xdg_wm_base_get_xdg_surface(xshell, wsurface);
    xdg_surface_add_listener(xshell_surface, &shell_surface_listener, this);
    xshell_toplevel = xdg_surface_get_toplevel(xshell_surface);
    xdg_toplevel_add_listener(xshell_toplevel, &toplevel_listener, this);
    xdg_toplevel_set_title(xshell_toplevel, title.c_str());
    xdg_toplevel_set_app_id(xshell_toplevel, title.c_str());
#else
    wshell_surface = wl_shell_get_shell_surface(wshell, wsurface);
    wl_shell_surface_add_listener(wshell_surface, &shell_surface_listener, this);
    wl_shell_surface_set_toplevel(wshell_surface);
    wl_shell_surface_set_title(wshell_surface, title.c_str());
    wl_shell_surface_set_class(wshell_surface, title.c_str());
#endif

    wl_display_sync(wdisplay);

    // construct window decoration
    const pangolin::Colour grey(0.5f, 0.5f, 0.5f, 0.5f);
    decoration = std::unique_ptr<Decoration>(new Decoration(5, 20, grey, wcompositor, wsubcompositor, wsurface, egl_display, egl_configs[0]));
    decoration->create();
    decoration->resize(width, height);
}

WaylandDisplay::~WaylandDisplay() {
    if(decoration)  decoration->destroy();

    // cleanup EGL
    if(egl_context) eglDestroyContext(egl_display, egl_context);
    if(egl_surface) eglDestroySurface(egl_display, egl_surface);
    if(egl_window)  wl_egl_window_destroy(egl_window);
    if(egl_display) eglTerminate(egl_display);

    // cleanup Wayland
#if USE_WL_XDG
    if(xshell_surface)  xdg_surface_destroy(xshell_surface);
    if(xshell_toplevel) xdg_toplevel_destroy(xshell_toplevel);
    if(xshell)          xdg_wm_base_destroy(xshell);
#else
    if(wshell_surface)  wl_shell_surface_destroy(wshell_surface);
    if(wshell)          wl_shell_destroy(wshell);
#endif

    if(wsurface)        wl_surface_destroy(wsurface);
    if(wregistry)       wl_registry_destroy(wregistry);
    if(wdisplay)        wl_display_disconnect(wdisplay);

    if(xkb_context) xkb_context_unref(xkb_context);
}

WaylandWindow::WaylandWindow(const int w, const int h, std::unique_ptr<WaylandDisplay> display) : display(std::move(display)){
    windowed_size[0] = w;
    windowed_size[1] = h;
}

WaylandWindow::~WaylandWindow() { }

void WaylandWindow::MakeCurrent() {
    eglMakeCurrent(display->egl_display, display->egl_surface, display->egl_surface, display->egl_context);
    context = this;
}

void WaylandWindow::RemoveCurrent() {
    eglMakeCurrent(display->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

void WaylandWindow::ToggleFullscreen() {
    is_fullscreen = !is_fullscreen;         // state for Pangolin
    display->is_fullscreen = is_fullscreen; // state for Wayland
    if(is_fullscreen) {
        display->decoration->destroy();
#if USE_WL_XDG
        xdg_toplevel_set_fullscreen(display->xshell_toplevel, nullptr);
#else
        wl_shell_surface_set_fullscreen(display->wshell_surface, WL_SHELL_SURFACE_FULLSCREEN_METHOD_DEFAULT, 0, nullptr);
#endif
    }
    else {
        display->decoration->create();
#if USE_WL_XDG
        xdg_toplevel_unset_fullscreen(display->xshell_toplevel);
#else
        wl_shell_surface_set_toplevel(display->wshell_surface);
#endif

#if !USE_WL_XDG
        if(display->is_maximised) {
            wl_shell_surface_set_maximized(display->wshell_surface, nullptr);
        }
        else {
            handle_configure(static_cast<void*>(display.get()), nullptr, 0,
                             windowed_size[0]+2*display->decoration->border_size,
                             windowed_size[1]+2*display->decoration->border_size+display->decoration->title_size);
        }
#endif
    }

    wl_display_sync(display->wdisplay);
}

void WaylandWindow::Move(const int /*x*/, const int /*y*/) { }

void WaylandWindow::Resize(const unsigned int /*w*/, const unsigned int /*h*/) { }

void WaylandWindow::ProcessEvents() { }

void WaylandWindow::SwapBuffers() {
    eglSwapBuffers(display->egl_display, display->egl_surface);

    // draw all decoration elements
    display->decoration->draw();

    MakeCurrent();

    wl_display_roundtrip(display->wdisplay);
}


std::unique_ptr<WindowInterface> CreateWaylandWindowAndBind(const std::string window_title, const int w, const int h, const std::string /*display_name*/, const bool /*double_buffered*/, const int /*sample_buffers*/, const int /*samples*/) {

    try{
        std::unique_ptr<WaylandDisplay> newdisplay = std::unique_ptr<WaylandDisplay>(new WaylandDisplay(w, h, window_title));

        // glewInit() fails with SIGSEGV for glew < 2.0 since it links to GLX
        if(atoi((char*)glewGetString(GLEW_VERSION_MAJOR))<2)
            return nullptr;

        WaylandWindow* win = new WaylandWindow(w, h, std::move(newdisplay));

        return std::unique_ptr<WindowInterface>(win);
    }
    catch(const std::runtime_error&) {
        // return null pointer for fallback to X11
        return nullptr;
    }
}

} // namespace wayland

PANGOLIN_REGISTER_FACTORY(WaylandWindow)
{
  struct WaylandWindowFactory : public FactoryInterface<WindowInterface> {
    std::unique_ptr<WindowInterface> Open(const Uri& uri) override {

      const std::string window_title = uri.Get<std::string>("window_title", "window");
      const int w = uri.Get<int>("w", 640);
      const int h = uri.Get<int>("h", 480);
      const std::string display_name = uri.Get<std::string>("display_name", "");
      const bool double_buffered = uri.Get<bool>("double_buffered", true);
      const int sample_buffers = uri.Get<int>("sample_buffers", 1);
      const int samples = uri.Get<int>("samples", 1);
      return std::unique_ptr<WindowInterface>(wayland::CreateWaylandWindowAndBind(window_title, w, h, display_name, double_buffered, sample_buffers, samples));
    }

    virtual ~WaylandWindowFactory() { }
  };

    auto factory = std::make_shared<WaylandWindowFactory>();
    FactoryRegistry<WindowInterface>::I().RegisterFactory(factory, 10, "wayland");
    FactoryRegistry<WindowInterface>::I().RegisterFactory(factory, 9,  "linux");
    FactoryRegistry<WindowInterface>::I().RegisterFactory(factory, 90,  "default");
}

} // namespace pangolin

