# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(Plotter main.cpp)
target_link_libraries(Plotter ${Pangolin_LIBRARIES})


# Make file association
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ${CMAKE_MAJOR_VERSION} VERSION_GREATER "2.9.9" )
    file( GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/pangoplot.desktop"
        CONTENT
"[Desktop Entry]
Version=1.0
Type=Application
Categories=Utility
Name=PangoPlotter
GenericName=Plotter
Comment=View CSV data series
Exec=$<TARGET_FILE:Plotter> %U
TryExec=$<TARGET_FILE:Plotter>
Icon=application-x-pango
Terminal=false
StartupNotify=false
MimeType=text/csv;text/comma-separated-values;application/x-pangoplot;application/csv"
    )

    add_custom_target(
        register-mime-info-plotter
        COMMAND mkdir -p $ENV{HOME}/.local/share/mime/packages/
        COMMAND mkdir -p $ENV{HOME}/.local/share/applications/
        COMMAND mkdir -p $ENV{HOME}/.local/share/icons/hicolor/scalable/mimetypes/
        COMMAND cp "${CMAKE_CURRENT_SOURCE_DIR}/application-x-pangoplot.xml" $ENV{HOME}/.local/share/mime/packages/
        COMMAND cp "${CMAKE_CURRENT_SOURCE_DIR}/../VideoViewer/application-x-pango.svg" $ENV{HOME}/.local/share/icons/hicolor/scalable/mimetypes/
        COMMAND cp "${CMAKE_CURRENT_BINARY_DIR}/pangoplot.desktop"       $ENV{HOME}/.local/share/applications/
        COMMAND gtk-update-icon-cache   $ENV{HOME}/.local/share/icons/hicolor -f -t
        COMMAND update-mime-database    $ENV{HOME}/.local/share/mime
        COMMAND update-desktop-database $ENV{HOME}/.local/share/applications
        DEPENDS Plotter
        VERBATIM
    )
endif()


#######################################################
## Install

install(TARGETS Plotter
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
