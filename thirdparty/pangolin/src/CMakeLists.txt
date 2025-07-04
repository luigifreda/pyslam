#######################################################
## Library sources

macro( append_glob list glob )
    file(GLOB files ${glob})
    set(${list} "${${list}};${files}")
endmacro()

## Header only includes / core headers
set( INCDIR "../include/pangolin" )
set( HEADERS
  ${INCDIR}/pangolin.h
  ${INCDIR}/platform.h
)
append_glob(HEADERS ${INCDIR}/utils/*.h*)
append_glob(HEADERS ${INCDIR}/image/*.h*)
append_glob(HEADERS ${INCDIR}/log/*.h*)
append_glob(HEADERS ${INCDIR}/geometry/*.h*)

### Store list of source files
append_glob(SOURCES utils/*.cpp)
append_glob(SOURCES image/*.cpp)
append_glob(SOURCES log/*.cpp)
append_glob(SOURCES geometry/*.cpp)

### Store list of Video factory registery methods to call for init.
include(CreateMethodCallFile)
set( VIDEO_FACTORY_REG "" )
set( WINDOW_FACTORY_REG "" )

#######################################################
## User build options

option(BUILD_PANGOLIN_GUI "Build support for Pangolin GUI" ON)
if(BUILD_PANGOLIN_GUI)
  append_glob(HEADERS ${INCDIR}/gl/*.h*)
  append_glob(HEADERS ${INCDIR}/display/*.h*)
  append_glob(HEADERS ${INCDIR}/handler/*.h*)

  append_glob(SOURCES gl/*.cpp)
  append_glob(SOURCES display/*.cpp)
  append_glob(SOURCES handler/*.cpp)

  if(NOT HAVE_GLES OR HAVE_GLES_2)
    append_glob(HEADERS ${INCDIR}/plot/*.h*)
    append_glob(SOURCES plot/*.cpp)
  endif()

endif()

option(BUILD_PANGOLIN_VARS "Build support for Pangolin Vars" ON)
if(BUILD_PANGOLIN_VARS)
  append_glob(HEADERS ${INCDIR}/var/*.h*)
  append_glob(SOURCES var/*.cpp)

  if(BUILD_PANGOLIN_GUI)
      list(APPEND HEADERS ${INCDIR}/display/widgets/widgets.h )
      list(APPEND SOURCES display/widgets/widgets.cpp )
  endif()
endif()

option(BUILD_PANGOLIN_VIDEO "Build support for Pangolin Video Utilities" ON)
if(BUILD_PANGOLIN_VIDEO)
  # Generic video includes
  append_glob(HEADERS ${INCDIR}/video/*.h*)
  append_glob(SOURCES video/*.cpp)

  # Generic video drivers
  list(APPEND HEADERS
    ${INCDIR}/video/drivers/test.h
    ${INCDIR}/video/drivers/images.h
    ${INCDIR}/video/drivers/images_out.h
    ${INCDIR}/video/drivers/split.h
    ${INCDIR}/video/drivers/truncate.h
    ${INCDIR}/video/drivers/pvn.h
    ${INCDIR}/video/drivers/pango.h
    ${INCDIR}/video/drivers/pango_video_output.h
    ${INCDIR}/video/drivers/debayer.h
    ${INCDIR}/video/drivers/shift.h
    ${INCDIR}/video/drivers/mirror.h
    ${INCDIR}/video/drivers/unpack.h
    ${INCDIR}/video/drivers/pack.h
    ${INCDIR}/video/drivers/join.h
    ${INCDIR}/video/drivers/merge.h
    ${INCDIR}/video/drivers/thread.h
  )
  list(APPEND SOURCES
    video/drivers/test.cpp
    video/drivers/images.cpp
    video/drivers/images_out.cpp
    video/drivers/split.cpp
    video/drivers/truncate.cpp
    video/drivers/pvn.cpp
    video/drivers/pango.cpp
    video/drivers/pango_video_output.cpp
    video/drivers/debayer.cpp
    video/drivers/shift.cpp
    video/drivers/mirror.cpp
    video/drivers/unpack.cpp
    video/drivers/pack.cpp
    video/drivers/join.cpp
    video/drivers/merge.cpp
    video/drivers/json.cpp
    video/drivers/thread.cpp
  )

  list(APPEND VIDEO_FACTORY_REG
    RegisterTestVideoFactory
    RegisterImagesVideoFactory
    RegisterImagesVideoOutputFactory
    RegisterSplitVideoFactory
    RegisterTruncateVideoFactory
    RegisterPvnVideoFactory
    RegisterPangoVideoFactory
    RegisterPangoVideoOutputFactory
    RegisterDebayerVideoFactory
    RegisterShiftVideoFactory
    RegisterMirrorVideoFactory
    RegisterUnpackVideoFactory
    RegisterPackVideoFactory
    RegisterJoinVideoFactory
    RegisterMergeVideoFactory
    RegisterJsonVideoFactory
    RegisterThreadVideoFactory
  )

  if(_LINUX_)
    list(APPEND HEADERS ${INCDIR}/video/drivers/shared_memory.h)
    list(APPEND SOURCES video/drivers/shared_memory.cpp)
    # Required for shared memory API using some versions of glibc
    list(APPEND LINK_LIBS rt pthread)
  endif()

endif()

if(BUILD_PANGOLIN_GUI AND BUILD_PANGOLIN_VARS AND BUILD_PANGOLIN_VIDEO )
    list(APPEND HEADERS ${INCDIR}/tools/video_viewer.h)
    list(APPEND SOURCES tools/video_viewer.cpp)
endif()

if(_LINUX_)
    set(OpenGL_GL_PREFERENCE "GLVND")
endif()

#######################################################
## Setup required includes / link info

option(DISPLAY_X11 "X11 Window Interface" ON)
option(DISPLAY_WAYLAND "Wayland Window Interface" OFF)

if(BUILD_PANGOLIN_GUI)
    if( ANDROID  )
        # Android specific display code
        list(APPEND HEADERS ${INCDIR}/display/device/display_android.h )
        list(APPEND SOURCES display/device/display_android.cpp )
        list(APPEND WINDOW_FACTORY_REG RegisterAndroidWindowFactory)

        if(HAVE_GLES_2)
            list(APPEND LINK_LIBS "-lEGL;-lGLESv2" )
        else()
            list(APPEND LINK_LIBS "-lEGL;-lGLESv1_CM" )
        endif()
    elseif( IOS )
        list(APPEND LINK_LIBS "-framework OpenGLES" )
        list(APPEND HEADERS "${INCDIR}/ios/PangolinAppDelegate.h" "${INCDIR}/ios/PangolinViewController.h" )
        list(APPEND SOURCES "ios/PangolinAppDelegate.mm" "ios/PangolinViewController.mm" )
        list(APPEND WINDOW_FACTORY_REG RegisterIosWindowFactory)
    else()
        find_package(OpenGL REQUIRED)
        list(APPEND USER_INC  ${OPENGL_INCLUDE_DIR})
        list(APPEND LINK_LIBS ${OPENGL_LIBRARIES})

        if(NOT BUILD_EXTERN_GLEW)
            find_package(GLEW REQUIRED)
        endif()
        list(APPEND USER_INC  ${GLEW_INCLUDE_DIR})
        list(APPEND LINK_LIBS ${GLEW_LIBRARY})
    endif()

    if( HAVE_GLES_2 )
        # Add Pangolins backwards compat layer.
        list(APPEND HEADERS ${INCDIR}/gl2engine.h )
        list(APPEND SOURCES gl2engine.cpp)
    endif()

    # headless offscreen rendering via EGL
    find_package(OpenGL QUIET COMPONENTS EGL)
    if(OpenGL_EGL_FOUND)
        list(APPEND WINDOW_FACTORY_REG RegisterNoneWindowFactory)
        list(APPEND SOURCES display/device/display_headless.cpp)
        list(APPEND LINK_LIBS ${OPENGL_egl_LIBRARY} )
    endif()
endif()

#######################################################
## Find optional dependencies

if(ANDROID)
    # Fix issue with thread local storage on android.
    add_definitions(-fno-data-sections)
    list(APPEND LINK_LIBS android log)
elseif(IOS)
    # Nothing specific in here yet.
else()
    if(BUILD_PANGOLIN_GUI)
        if(_WIN_)
            list(APPEND WINDOW_FACTORY_REG RegisterWinWindowFactory)
            list(APPEND SOURCES display/device/display_win.cpp )
        elseif(_OSX_)
            list(APPEND WINDOW_FACTORY_REG RegisterOsxWindowFactory)
            list(APPEND SOURCES display/device/display_osx.mm display/device/PangolinNSApplication.mm display/device/PangolinNSGLView.mm  )
            list(APPEND LINK_LIBS "-framework Cocoa" )
        elseif(_LINUX_)
            # Wayland
            if(DISPLAY_WAYLAND)
                find_package(Wayland QUIET)
                if(WAYLAND_CLIENT_FOUND)
                    find_package(Wayland REQUIRED)
                    find_package(PkgConfig)
                    pkg_check_modules(xkbcommon REQUIRED xkbcommon)
                    pkg_check_modules(egl QUIET egl)

                    # find Wayland protocols
                    pkg_get_variable(WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)

                    # generate header for protocol 'xdg_shell'
                    set(XDG_PROT_DEF "${WAYLAND_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml")
                    if(EXISTS ${XDG_PROT_DEF})
                        # use the 'xdg_shell' protocol
                        add_definitions(-DUSE_WL_XDG=1)

                        # find 'wayland-scanner' executable
                        pkg_get_variable(WAYLAND_SCANNER wayland-scanner wayland_scanner)

                        # generate protocol implementation
                        add_custom_command(
                            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-client-protocol.h
                            COMMAND ${WAYLAND_SCANNER} client-header ${XDG_PROT_DEF} xdg-shell-client-protocol.h)
                        add_custom_command(
                            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-protocol.c
                            COMMAND ${WAYLAND_SCANNER} private-code ${XDG_PROT_DEF} xdg-shell-protocol.c
                            DEPENDS xdg-shell-client-protocol.h)
                        include_directories(${CMAKE_CURRENT_BINARY_DIR})
                        list(APPEND SOURCES xdg-shell-protocol.c)

                    else()
                        # use deprecated 'wl_shell' interface
                        add_definitions(-DUSE_WL_XDG=0)
                    endif()

                    list(APPEND WINDOW_FACTORY_REG RegisterWaylandWindowFactory)
                    list(APPEND SOURCES display/device/display_wayland.cpp)
                    list(APPEND LINK_LIBS
                        ${WAYLAND_CLIENT_LIBRARIES}
                        ${WAYLAND_EGL_LIBRARIES}
                        ${WAYLAND_CURSOR_LIBRARIES}
                        ${egl_LIBRARIES}
                        ${xkbcommon_LIBRARIES}
                    )
                    list(APPEND INTERNAL_INC
                        ${WAYLAND_CLIENT_INCLUDE_DIR}
                        ${WAYLAND_EGL_INCLUDE_DIR}
                        ${WAYLAND_CURSOR_INCLUDE_DIR}
                        ${egl_INCLUDE_DIRS}
                        ${xkbcommon_INCLUDE_DIRS}
                    )
                endif()
            endif()

            # X11
            find_package(X11 QUIET)
            if(DISPLAY_X11 AND X11_FOUND)
                list(APPEND WINDOW_FACTORY_REG RegisterX11WindowFactory)
                list(APPEND USER_INC ${X11_INCLUDE_DIR})
                list(APPEND SOURCES display/device/display_x11.cpp )
                list(APPEND LINK_LIBS ${X11_LIBRARIES} )
            endif()
        endif()
    endif()
endif()

if(UNIX)
  append_glob(HEADERS ${INCDIR}/utils/posix/*.h*)
  append_glob(SOURCES utils/posix/*.cpp)
  if(_LINUX_)
    # Required for shared memory API using some versions of glibc
    list(APPEND LINK_LIBS rt pthread)
  endif()
endif()

option(BUILD_PANGOLIN_PYTHON "Build support for Pangolin Interactive Console" OFF)
if(BUILD_PANGOLIN_PYTHON AND BUILD_PANGOLIN_GUI AND BUILD_PANGOLIN_VARS AND NOT _WIN_)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../external/pybind11/CMakeLists.txt")
      add_subdirectory("../external/pybind11" "${CMAKE_CURRENT_BINARY_DIR}/external/pybind11")
      set( pybind11_FOUND true)
  else()
      find_package(pybind11 QUIET)
  endif()

  if(pybind11_FOUND)
    set(HAVE_PYTHON 1)

    file(GLOB pypangolin_SRC "python/pypangolin/*.cpp" )
    file(GLOB pypangolin_HDR "python/pypangolin/*.hpp" )
    list(APPEND HEADERS
        ${INCDIR}/console/ConsoleInterpreter.h
        ${INCDIR}/console/ConsoleView.h
        ${INCDIR}/python/pyinterpreter.h
        ${INCDIR}/python/pypangolin_init.h
        ${INCDIR}/python/pyuniqueobj.h
        ${INCDIR}/python/pyvar.h
        ${INCDIR}/python/pypangoio.h
        ${pypangolin_HDR}
    )
    list(APPEND SOURCES
        console/ConsoleView.cpp
        python/pyinterpreter.cpp
        python/pypangolin_init.cpp
        ${pypangolin_SRC}
    )

    # pybind11 stuff
    list(APPEND INTERNAL_INC ${PYBIND11_INCLUDE_DIR})
    list(APPEND LINK_LIBS PRIVATE ${pybind11_LIBRARIES})

    if(_GCC_)
        set_source_files_properties(python/PyInterpreter.cpp    PROPERTIES COMPILE_FLAGS "-Wno-missing-field-initializers")
        set_source_files_properties(python/PyModulePangolin.cpp PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing -Wno-missing-field-initializers")
    endif()
    list(APPEND INTERNAL_INC ${PYTHON_INCLUDE_DIRS})
    list(APPEND LINK_LIBS    ${PYTHON_LIBRARIES})
    message(STATUS "Python Found and Enabled (with pybind11)")
  endif()
endif()

option(BUILD_PANGOLIN_EIGEN "Build support for Eigen matrix types" ON)
if(BUILD_PANGOLIN_EIGEN)
  find_package(Eigen QUIET)
  if(EIGEN_FOUND)
    set(HAVE_EIGEN 1)
    list(APPEND USER_INC  ${EIGEN_INCLUDE_DIR} )
    if(_CLANG_)
      # Eigen causes many of these errors. Suppress.
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
    endif()
    message(STATUS "Eigen Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_TOON "Build support for TooN matrix types" ON)
if(BUILD_PANGOLIN_TOON)
  find_package(TooN QUIET)
  if(TooN_FOUND)
    set(HAVE_TOON 1)
    list(APPEND USER_INC  ${TooN_INCLUDE_DIR} )
    message(STATUS "TooN Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBDC1394 "Build support for libdc1394 video input" ON)
if(BUILD_PANGOLIN_LIBDC1394 AND BUILD_PANGOLIN_VIDEO)
  find_package(DC1394 QUIET)
  if(DC1394_FOUND)
    set(HAVE_DC1394 1)
    list(APPEND INTERNAL_INC  ${DC1394_INCLUDE_DIR} )
    list(APPEND LINK_LIBS  ${DC1394_LIBRARY} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/firewire.h ${INCDIR}/video/drivers/deinterlace.h )
    list(APPEND SOURCES video/drivers/firewire.cpp video/drivers/deinterlace.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterFirewireVideoFactory )
    message(STATUS "libdc1394 Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_V4L "Build support for V4L video input" ON)
if(BUILD_PANGOLIN_V4L AND BUILD_PANGOLIN_VIDEO AND _LINUX_)
  set(HAVE_V4L 1)
  list(APPEND HEADERS ${INCDIR}/video/drivers/v4l.h)
  list(APPEND SOURCES video/drivers/v4l.cpp)
  list(APPEND VIDEO_FACTORY_REG RegisterV4lVideoFactory )
  message(STATUS "V4L Found and Enabled")
endif()

option(BUILD_PANGOLIN_FFMPEG "Build support for ffmpeg video input" ON)
if(BUILD_PANGOLIN_FFMPEG AND BUILD_PANGOLIN_VIDEO)
  find_package(FFMPEG QUIET)
  if(FFMPEG_FOUND)
    set(HAVE_FFMPEG 1)
    list(APPEND INTERNAL_INC  ${FFMPEG_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${FFMPEG_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/ffmpeg.h)
    list(APPEND SOURCES video/drivers/ffmpeg.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterFfmpegVideoFactory )
    list(APPEND VIDEO_FACTORY_REG RegisterFfmpegVideoOutputFactory )

    if(_GCC_)
      # FFMPEG is a real pain for deprecating the API.
      set_source_files_properties(video/drivers/ffmpeg.cpp    PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    endif()
    message(STATUS "ffmpeg Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBREALSENSE "Build support for LibRealSense video input" ON)
if(BUILD_PANGOLIN_LIBREALSENSE AND BUILD_PANGOLIN_VIDEO)
  find_package(LibRealSense QUIET)
  if(LIBREALSENSE_FOUND)
    set(HAVE_LIBREALSENSE 1)
    list(APPEND INTERNAL_INC  ${LIBREALSENSE_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${LIBREALSENSE_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/realsense.h )
    list(APPEND SOURCES video/drivers/realsense.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterRealSenseVideoFactory )
    message(STATUS "LibRealSense Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBREALSENSE2 "Build support for LibRealSense2 video input" ON)
if(BUILD_PANGOLIN_LIBREALSENSE2 AND BUILD_PANGOLIN_VIDEO)
  find_package(LibRealSense2 QUIET)
  if(LIBREALSENSE2_FOUND)
    set(HAVE_LIBREALSENSE2 1)
    list(APPEND INTERNAL_INC  ${LIBREALSENSE2_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${LIBREALSENSE2_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/realsense2.h )
    list(APPEND SOURCES video/drivers/realsense2.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterRealSense2VideoFactory )
    message(STATUS "LibRealSense2 Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_OPENNI "Build support for OpenNI video input" ON)
if(BUILD_PANGOLIN_OPENNI AND BUILD_PANGOLIN_VIDEO)
  find_package(OpenNI QUIET)
  if(OPENNI_FOUND)
    set(HAVE_OPENNI 1)
    if(_LINUX_)
      add_definitions(-Dlinux=1)
    endif()
    list(APPEND INTERNAL_INC  ${OPENNI_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${OPENNI_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/openni.h )
    list(APPEND SOURCES video/drivers/openni.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterOpenNiVideoFactory )
    message(STATUS "OpenNI Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_OPENNI2 "Build support for OpenNI2 video input" ON)
if(BUILD_PANGOLIN_OPENNI2 AND BUILD_PANGOLIN_VIDEO)
  find_package(OpenNI2 QUIET)
  if(OPENNI2_FOUND)
    set(HAVE_OPENNI2 1)
    if(_LINUX_)
      add_definitions(-Dlinux=1)
    endif()
    list(APPEND INTERNAL_INC  ${OPENNI2_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${OPENNI2_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/openni2.h )
    list(APPEND SOURCES video/drivers/openni2.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterOpenNi2VideoFactory )
    message(STATUS "OpenNI2 Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBUVC "Build support for libuvc video input" ON)
if(BUILD_PANGOLIN_LIBUVC AND BUILD_PANGOLIN_VIDEO)
  find_package(uvc QUIET)
  if(uvc_FOUND)
    set(HAVE_UVC 1)
    list(APPEND INTERNAL_INC  ${uvc_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${uvc_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/uvc.h )
    list(APPEND SOURCES video/drivers/uvc.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterUvcVideoFactory )
    if(_WIN_)
      find_package(pthread REQUIRED QUIET)
      list(APPEND LINK_LIBS ${pthread_LIBRARIES} )

      find_package(libusb1 REQUIRED QUIET)
      list(APPEND LINK_LIBS ${libusb1_LIBRARIES} )
    endif()
    message(STATUS "libuvc Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_UVC_MEDIAFOUNDATION "Build support for MediaFoundation UVC input" ON)
if (BUILD_PANGOLIN_UVC_MEDIAFOUNDATION AND BUILD_PANGOLIN_VIDEO)
  find_package(MediaFoundation QUIET)
  if (MediaFoundation_FOUND)
    list(APPEND LINK_LIBS ${MediaFoundation_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/uvc_mediafoundation.h )
    list(APPEND SOURCES video/drivers/uvc_mediafoundation.cpp )
    list(APPEND VIDEO_FACTORY_REG RegisterUvcMediaFoundationVideoFactory )
    message(STATUS "MediaFoundation Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_DEPTHSENSE "Build support for DepthSense video input" ON)
if(BUILD_PANGOLIN_DEPTHSENSE AND BUILD_PANGOLIN_VIDEO)
  find_package(DepthSense QUIET)
  if(DepthSense_FOUND)
    set(HAVE_DEPTHSENSE 1)
    list(APPEND INTERNAL_INC  ${DepthSense_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${DepthSense_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/depthsense.h )
    list(APPEND SOURCES video/drivers/depthsense.cpp)
    list(APPEND VIDEO_FACTORY_REG RegisterDepthSenseVideoFactory )
    message(STATUS "DepthSense Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_TELICAM "Build support for TeliCam video input" ON)
if(BUILD_PANGOLIN_TELICAM AND BUILD_PANGOLIN_VIDEO)
  find_package(TeliCam QUIET)
  if(TeliCam_FOUND)
    set(HAVE_TELICAM 1)
    list(APPEND INTERNAL_INC ${TeliCam_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${TeliCam_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/teli.h )
    list(APPEND SOURCES video/drivers/teli.cpp )
    list(APPEND VIDEO_FACTORY_REG RegisterTeliVideoFactory )

    message(STATUS "TeliCam Found and Enabled" )
  endif()
endif()

option(BUILD_PANGOLIN_PLEORA "Build support for Pleora video input" ON)
if(BUILD_PANGOLIN_PLEORA AND BUILD_PANGOLIN_VIDEO)
  find_package(Pleora QUIET)
  if(Pleora_FOUND)
    set(HAVE_PLEORA 1)
    list(APPEND INTERNAL_INC ${Pleora_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${Pleora_LIBRARIES} )
    list(APPEND HEADERS ${INCDIR}/video/drivers/pleora.h )
    list(APPEND SOURCES video/drivers/pleora.cpp )
    list(APPEND VIDEO_FACTORY_REG RegisterPleoraVideoFactory )

    if(_GCC_)
      # Suppress warnings generated from Pleora SDK.
      set_source_files_properties(video/drivers/pleora.cpp PROPERTIES COMPILE_FLAGS -Wno-unknown-pragmas)
      set_source_files_properties(video/video.cpp PROPERTIES COMPILE_FLAGS -Wno-unknown-pragmas)
    endif()
    message(STATUS "Pleora Found and Enabled" )
  endif()
endif()

option(BUILD_PANGOLIN_LIBPNG "Build support for libpng image input" ON)
if(BUILD_PANGOLIN_LIBPNG)
  if(NOT BUILD_EXTERN_LIBPNG)
    find_package(PNG QUIET)
  endif()
  if(PNG_FOUND)
    # (ZLIB is also found by FindPNG.cmake as its dependency)
    set(HAVE_PNG 1)
    list(APPEND INTERNAL_INC ${PNG_INCLUDE_DIR} )
    list(APPEND LINK_LIBS ${PNG_LIBRARY} ${ZLIB_LIBRARY} )
    message(STATUS "libpng Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBJPEG "Build support for libjpeg image input" ON)
if(BUILD_PANGOLIN_LIBJPEG)
  if(NOT BUILD_EXTERN_LIBJPEG)
    find_package(JPEG QUIET)
  endif()
  if(JPEG_FOUND)
    set(HAVE_JPEG 1)
    list(APPEND INTERNAL_INC ${JPEG_INCLUDE_DIR} )
    list(APPEND LINK_LIBS ${JPEG_LIBRARY} )
    message(STATUS "libjpeg Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBTIFF "Build support for libtiff image input" ON)
if(BUILD_PANGOLIN_LIBTIFF)
  find_package(TIFF QUIET)
  if(TIFF_FOUND)
    set(HAVE_TIFF 1)
    list(APPEND INTERNAL_INC ${TIFF_INCLUDE_DIR} )
    list(APPEND LINK_LIBS ${TIFF_LIBRARY} )
    message(STATUS "libtiff Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LIBOPENEXR "Build support for libopenexr image input" ON)
if(BUILD_PANGOLIN_LIBOPENEXR)
  find_package(OpenEXR QUIET)
  if(OpenEXR_FOUND)
    set(HAVE_OPENEXR 1)
    list(APPEND INTERNAL_INC ${OpenEXR_INCLUDE_DIR} )
    list(APPEND LINK_LIBS ${OpenEXR_LIBRARY} )
    message(STATUS "libopenexr Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_ZSTD "Build support for libzstd compression" ON)
if(BUILD_PANGOLIN_ZSTD)
  find_package(zstd QUIET)
  if(zstd_FOUND)
    set(HAVE_ZSTD 1)
    list(APPEND INTERNAL_INC ${zstd_INCLUDE_DIR} )
    list(APPEND LINK_LIBS ${zstd_LIBRARY} )
    message(STATUS "libzstd Found and Enabled")
  endif()
endif()

option(BUILD_PANGOLIN_LZ4 "Build support for liblz4 compression" ON)
if(BUILD_PANGOLIN_LZ4)
  find_package(Lz4 QUIET)
  if(Lz4_FOUND)
    set(HAVE_LZ4 1)
    list(APPEND INTERNAL_INC ${Lz4_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${Lz4_LIBRARIES} )
    message(STATUS "liblz4 Found and Enabled")
  endif()
endif()

#######################################################
## Embed resource binary files

include(EmbedBinaryFiles)

if(BUILD_PANGOLIN_GUI)
    embed_binary_files( "_embed_/fonts/*.ttf" "${CMAKE_CURRENT_BINARY_DIR}/fonts.cpp" )
    list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/fonts.cpp" )
endif()

#######################################################
## Add Libraries / Include Directories / Link directories

set(INSTALL_INCLUDE_DIR "include")

add_library(${LIBRARY_NAME} ${SOURCES} ${HEADERS})
# 'System' includes shield us from warnings in those includes.
target_include_directories(${LIBRARY_NAME} SYSTEM PUBLIC ${USER_INC} PRIVATE ${INTERNAL_INC})
target_include_directories(${LIBRARY_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                                                  $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)
target_link_libraries(${LIBRARY_NAME} PUBLIC ${LINK_LIBS})

## Generate symbol export helper header on MSVC
IF(MSVC)
    string(TOUPPER ${LIBRARY_NAME} LIBRARY_NAME_CAPS)
    include(GenerateExportHeader)
    GENERATE_EXPORT_HEADER( ${LIBRARY_NAME}
        BASE_NAME ${LIBRARY_NAME_CAPS}
        EXPORT_MACRO_NAME ${LIBRARY_NAME_CAPS}_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/${LIBRARY_NAME}_export.h"
        STATIC_DEFINE ${LIBRARY_NAME_CAPS}_BUILT_AS_STATIC
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/${LIBRARY_NAME}_export.h"
      DESTINATION ${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}/${LIBRARY_NAME}
    )
ENDIF()

## Set Special Compiler flags
if(MSVC)
  set(CMAKE_CXX_FLAGS "/EHs ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-Wall -Wno-error=deprecated-declarations ${CMAKE_CXX_FLAGS}")
endif()

#######################################################
## Create config.h file for inclusion in library

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/config.h"
)

#######################################################
## Create video_drivers.h file for inclusion in library
## This loads video driver factories based on cmake configuration

CreateMethodCallFile(
    "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/video_drivers.h"
    "pangolin" "LoadBuiltInVideoDrivers" "${VIDEO_FACTORY_REG}"
)

#######################################################
## Create window_frameworks.h file for inclusion in library
## This loads windowing framwork factories based on cmake configuration

CreateMethodCallFile(
    "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/window_frameworks.h"
    "pangolin" "LoadBuiltInWindowFrameworks" "${WINDOW_FACTORY_REG}"
)

#######################################################
## Python wrapper

option(BUILD_PYPANGOLIN_MODULE "Python wrapper for Pangolin" ON)
if(BUILD_PYPANGOLIN_MODULE AND HAVE_PYTHON )
    file(GLOB pypangolin_SRC "python/pypangolin/*.hpp" "python/pypangolin/*.cpp" "python/pypangolin_module.cpp")
    pybind11_add_module(pypangolin ${pypangolin_SRC})
    target_link_libraries(pypangolin PRIVATE ${LIBRARY_NAME})
    target_include_directories(pypangolin PRIVATE "${USER_INC}")
endif()

#######################################################
## Generate Doxygen documentation target (make pangolin_doc)
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../doc )
    add_custom_target(pangolin_doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../doc
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif()

#######################################################

# This relative path allows installed files to be relocatable.
set( CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME} )
file( RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
    "${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}"
)

# Export library for easy inclusion from other cmake projects. APPEND allows
# call to function even as subdirectory of larger project.
FILE(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export( TARGETS ${LIBRARY_NAME}
        APPEND FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake" )

# Version information
configure_file(${PROJECT_NAME}ConfigVersion.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Build tree config
set( EXPORT_LIB_INC_DIR "${PROJECT_SOURCE_DIR}/include;${CMAKE_CURRENT_BINARY_DIR}/include" )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE )

# Install tree config
set( EXPORT_LIB_INC_DIR "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake @ONLY )

# Add package to CMake package registery for use from the build tree
option( EXPORT_${PROJECT_NAME}
  "Should the ${PROJECT_NAME} package be exported for use by other software" ON )

if( EXPORT_${PROJECT_NAME} )
  export( PACKAGE ${PROJECT_NAME} )
endif()

#######################################################
## Install headers / targets

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/config.h"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}/${LIBRARY_NAME}
)
install(DIRECTORY ${INCDIR}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}
)
install(TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

#######################################################
## Install CMake config

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)
install(
    EXPORT ${PROJECT_NAME}Targets DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)
