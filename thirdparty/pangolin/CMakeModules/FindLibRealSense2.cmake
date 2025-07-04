# -*- mode: cmake; -*-
###############################################################################
# Find librealsense2  https://github.com/IntelRealSense/librealsense
#
# This sets the following variables:
# LIBREALSENSE2_FOUND - True if LIBREALSENSE2 was found.
# LIBREALSENSE2_INCLUDE_DIRS - Directories containing the LIBREALSENSE2 include files.
# LIBREALSENSE2_LIBRARIES - Libraries needed to use LIBREALSENSE2.
# LIBREALSENSE2_DEFINITIONS - Compiler flags for LIBREALSENSE2.
#
# File forked from augmented_dev, project of alantrrs
# (https://github.com/alantrrs/augmented_dev).

find_package(PkgConfig)
if(${CMAKE_VERSION} VERSION_LESS 2.8.2)
endif()

#add a hint so that it can find it without the pkg-config
find_path(LIBREALSENSE2_INCLUDE_DIR librealsense2/rs.h
    HINTS /usr/include/  /usr/local/include)
#add a hint so that it can find it without the pkg-config
find_library(LIBREALSENSE2_LIBRARY
    NAMES librealsense2.so
    HINTS /usr/lib /usr/local/lib )

  set(LIBREALSENSE2_INCLUDE_DIRS ${LIBREALSENSE2_INCLUDE_DIR})
  set(LIBREALSENSE2_LIBRARIES ${LIBREALSENSE2_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibRealSense2 DEFAULT_MSG
  LIBREALSENSE2_LIBRARY LIBREALSENSE2_INCLUDE_DIR)

mark_as_advanced(LIBREALSENSE2_LIBRARY LIBREALSENSE2_INCLUDE_DIR)
