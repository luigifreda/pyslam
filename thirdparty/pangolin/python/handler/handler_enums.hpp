#include <pybind11/pybind11.h>

#include <pangolin/handler/handler_enums.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

enum PangoKey {
    // Supported Key modifiers for GlobalKeyPressCallback.
    // e.g. PANGO_CTRL + 'r', PANGO_SPECIAL + GLUT_KEY_RIGHT, etc.
    PANGO_SPECIAL_ = PANGO_SPECIAL,
    PANGO_CTRL_ = PANGO_CTRL,
    PANGO_OPTN_ = PANGO_OPTN,

    // Ordinary keys
    PANGO_KEY_TAB_ = PANGO_KEY_TAB,
    PANGO_KEY_ESCAPE_ = PANGO_KEY_ESCAPE,

    // Special Keys (same as GLUT_ defines)
    PANGO_KEY_F1_ = PANGO_KEY_F1,
    PANGO_KEY_F2_ = PANGO_KEY_F2,
    PANGO_KEY_F3_ = PANGO_KEY_F3,
    PANGO_KEY_F4_ = PANGO_KEY_F4,
    PANGO_KEY_F5_ = PANGO_KEY_F5,
    PANGO_KEY_F6_ = PANGO_KEY_F6,
    PANGO_KEY_F7_ = PANGO_KEY_F7,
    PANGO_KEY_F8_ = PANGO_KEY_F8,
    PANGO_KEY_F9_ = PANGO_KEY_F9,
    PANGO_KEY_F10_ = PANGO_KEY_F10,
    PANGO_KEY_F11_ = PANGO_KEY_F11,
    PANGO_KEY_F12_ = PANGO_KEY_F12,
    PANGO_KEY_LEFT_ = PANGO_KEY_LEFT,
    PANGO_KEY_UP_ = PANGO_KEY_UP,
    PANGO_KEY_RIGHT_ = PANGO_KEY_RIGHT,
    PANGO_KEY_DOWN_ = PANGO_KEY_DOWN,
    PANGO_KEY_PAGE_UP_ = PANGO_KEY_PAGE_UP,
    PANGO_KEY_PAGE_DOWN_ = PANGO_KEY_PAGE_DOWN,
    PANGO_KEY_HOME_ = PANGO_KEY_HOME,
    PANGO_KEY_END_ = PANGO_KEY_END,
    PANGO_KEY_INSERT_ = PANGO_KEY_INSERT,
};



void declareHandlerEnums(py::module & m) {

    py::enum_<PangoKey>(m, "PangoKey")
        .value("PANGO_SPECIAL", PangoKey::PANGO_SPECIAL_)
        .value("PANGO_CTRL", PangoKey::PANGO_CTRL_)
        .value("PANGO_OPTN", PangoKey::PANGO_OPTN_)

        .value("PANGO_KEY_TAB", PangoKey::PANGO_KEY_TAB_)
        .value("PANGO_KEY_ESCAPE", PangoKey::PANGO_KEY_ESCAPE_)

        .value("PANGO_KEY_F1", PangoKey::PANGO_KEY_F1_)
        .value("PANGO_KEY_F2", PangoKey::PANGO_KEY_F2_)
        .value("PANGO_KEY_F3", PangoKey::PANGO_KEY_F3_)
        .value("PANGO_KEY_F4", PangoKey::PANGO_KEY_F4_)
        .value("PANGO_KEY_F5", PangoKey::PANGO_KEY_F5_)
        .value("PANGO_KEY_F6", PangoKey::PANGO_KEY_F6_)
        .value("PANGO_KEY_F7", PangoKey::PANGO_KEY_F7_)
        .value("PANGO_KEY_F8", PangoKey::PANGO_KEY_F8_)
        .value("PANGO_KEY_F9", PangoKey::PANGO_KEY_F9_)
        .value("PANGO_KEY_F10", PangoKey::PANGO_KEY_F10_)
        .value("PANGO_KEY_F11", PangoKey::PANGO_KEY_F11_)
        .value("PANGO_KEY_F12", PangoKey::PANGO_KEY_F12_)
        .value("PANGO_KEY_LEFT", PangoKey::PANGO_KEY_LEFT_)
        .value("PANGO_KEY_UP", PangoKey::PANGO_KEY_UP_)
        .value("PANGO_KEY_RIGHT", PangoKey::PANGO_KEY_RIGHT_)
        .value("PANGO_KEY_DOWN", PangoKey::PANGO_KEY_DOWN_)
        .value("PANGO_KEY_PAGE_UP", PangoKey::PANGO_KEY_PAGE_UP_)
        .value("PANGO_KEY_PAGE_DOWN", PangoKey::PANGO_KEY_PAGE_DOWN_)
        .value("PANGO_KEY_HOME", PangoKey::PANGO_KEY_HOME_)
        .value("PANGO_KEY_END", PangoKey::PANGO_KEY_END_)
        .value("PANGO_KEY_INSERT", PangoKey::PANGO_KEY_INSERT_)
        .export_values();

    
        /*m.def_readonly("PANGO_SPECIAL", &PANGO_SPECIAL);  // py::module no def_readonly
        m.def_readonly("PANGO_CTRL", &PANGO_CTRL);
        m.def_readonly("PANGO_OPTN", &PANGO_OPTN);

        m.def_readonly("PANGO_KEY_TAB", &PANGO_KEY_TAB);
        m.def_readonly("PANGO_KEY_ESCAPE", &PANGO_KEY_ESCAPE);

        m.def_readonly("PANGO_KEY_F1", &PANGO_KEY_F1);
        m.def_readonly("PANGO_KEY_F2", &PANGO_KEY_F2);
        m.def_readonly("PANGO_KEY_F3", &PANGO_KEY_F3);
        m.def_readonly("PANGO_KEY_F4", &PANGO_KEY_F4);
        m.def_readonly("PANGO_KEY_F5", &PANGO_KEY_F5);
        m.def_readonly("PANGO_KEY_F6", &PANGO_KEY_F6);
        m.def_readonly("PANGO_KEY_F7", &PANGO_KEY_F7);
        m.def_readonly("PANGO_KEY_F8", &PANGO_KEY_F8);
        m.def_readonly("PANGO_KEY_F9", &PANGO_KEY_F9);
        m.def_readonly("PANGO_KEY_F10", &PANGO_KEY_F10);
        m.def_readonly("PANGO_KEY_F11", &PANGO_KEY_F11);
        m.def_readonly("PANGO_KEY_F12", &PANGO_KEY_F12);
        m.def_readonly("PANGO_KEY_LEFT", &PANGO_KEY_LEFT);
        m.def_readonly("PANGO_KEY_UP", &PANGO_KEY_UP);
        m.def_readonly("PANGO_KEY_RIGHT", &PANGO_KEY_RIGHT);
        m.def_readonly("PANGO_KEY_DOWN", &PANGO_KEY_DOWN);
        m.def_readonly("PANGO_KEY_PAGE_UP", &PANGO_KEY_PAGE_UP);
        m.def_readonly("PANGO_KEY_PAGE_DOWN", &PANGO_KEY_PAGE_DOWN);
        m.def_readonly("PANGO_KEY_HOME", &PANGO_KEY_HOME);
        m.def_readonly("PANGO_KEY_END", &PANGO_KEY_END);
        m.def_readonly("PANGO_KEY_INSERT", &PANGO_KEY_INSERT);*/

    py::enum_<MouseButton>(m, "MouseButton")
        .value("MouseButtonLeft", MouseButton::MouseButtonLeft)
        .value("MouseButtonMiddle", MouseButton::MouseButtonMiddle)
        .value("MouseButtonRight", MouseButton::MouseButtonRight)
        .value("MouseWheelUp", MouseButton::MouseWheelUp)
        .value("MouseWheelDown", MouseButton::MouseWheelDown)
        .value("MouseWheelRight", MouseButton::MouseWheelRight)
        .value("MouseWheelLeft", MouseButton::MouseWheelLeft)
        .export_values();

    py::enum_<KeyModifier>(m, "KeyModifier")
        .value("KeyModifierShift", KeyModifier::KeyModifierShift)
        .value("KeyModifierCtrl", KeyModifier::KeyModifierCtrl)
        .value("KeyModifierAlt", KeyModifier::KeyModifierAlt)
        .value("KeyModifierCmd", KeyModifier::KeyModifierCmd)
        .value("KeyModifierFnc", KeyModifier::KeyModifierFnc)
        .export_values();

    py::enum_<InputSpecial>(m, "InputSpecial")
        .value("InputSpecialScroll", InputSpecial::InputSpecialScroll)
        .value("InputSpecialZoom", InputSpecial::InputSpecialZoom)
        .value("InputSpecialRotate", InputSpecial::InputSpecialRotate)
        .value("InputSpecialTablet", InputSpecial::InputSpecialTablet)
        .export_values();

}

}  // namespace pangolin::