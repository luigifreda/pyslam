include(ExternalProject)

set(ExternConfig "")

if( BUILD_EXTERN_GLEW )

#########################################################
# GLEW
#########################################################
ExternalProject_Add( __glew
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/glew"
  GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
  GIT_TAG 7574ab4d00b683e56adbfdec7da636529dfe65d8
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_MINSIZEREL=${CMAKE_C_FLAGS_MINSIZEREL}
             -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
)

add_library(_glew STATIC IMPORTED GLOBAL)
add_dependencies(_glew __glew)
set_target_properties(_glew PROPERTIES
    IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_INSTALL_PREFIX}/lib/glew.lib
    IMPORTED_LOCATION_RELEASE ${CMAKE_INSTALL_PREFIX}/lib/glew.lib
    IMPORTED_LOCATION_DEBUG   ${CMAKE_INSTALL_PREFIX}/lib/glewd.lib
)

set(GLEW_FOUND true PARENT_SCOPE)
set(GLEW_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include" PARENT_SCOPE)
set(GLEW_LIBRARY _glew PARENT_SCOPE)
set(GLEW_STATIC 1 PARENT_SCOPE)
set(ExternConfig "${ExternConfig}
  add_library(_glew STATIC IMPORTED)
  set_target_properties(_glew PROPERTIES
    IMPORTED_LOCATION_RELWITHDEBINFO \"${CMAKE_INSTALL_PREFIX}/lib/glew.lib\"
    IMPORTED_LOCATION_RELEASE \"${CMAKE_INSTALL_PREFIX}/lib/glew.lib\"
    IMPORTED_LOCATION_DEBUG   \"${CMAKE_INSTALL_PREFIX}/lib/glewd.lib\"
  )")
endif()

if( BUILD_EXTERN_LIBPNG )

#########################################################
# zlib
#########################################################

ExternalProject_Add( __zlib
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zlib"
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.8
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_MINSIZEREL=${CMAKE_C_FLAGS_MINSIZEREL}
             -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
)
add_library(_zlib STATIC IMPORTED GLOBAL)
add_dependencies(_zlib __zlib)
set_target_properties(_zlib PROPERTIES
    IMPORTED_LOCATION_RELEASE ${CMAKE_INSTALL_PREFIX}/lib/zlibstatic.lib
    IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_INSTALL_PREFIX}/lib/zlibstatic.lib
    IMPORTED_LOCATION_DEBUG   ${CMAKE_INSTALL_PREFIX}/lib/zlibstaticd.lib
)

#########################################################
# libpng
#########################################################

ExternalProject_Add( __libpng
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libpng"
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG v1.6.18
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
              -DZLIB_INCLUDE_DIR=${CMAKE_INSTALL_PREFIX}/include
              -DZLIB_LIBRARY=${CMAKE_INSTALL_PREFIX}/lib/zlibstatic*.lib
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
              -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
              -DCMAKE_C_FLAGS_MINSIZEREL=${CMAKE_C_FLAGS_MINSIZEREL}
              -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
  DEPENDS __zlib
)

add_library(_libpng STATIC IMPORTED GLOBAL)
add_dependencies(_libpng __libpng)
set_target_properties(_libpng PROPERTIES
    IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_INSTALL_PREFIX}/lib/libpng16_static.lib
	IMPORTED_LOCATION_RELEASE ${CMAKE_INSTALL_PREFIX}/lib/libpng16_static.lib
    IMPORTED_LOCATION_DEBUG   ${CMAKE_INSTALL_PREFIX}/lib/libpng16_staticd.lib
)

set(PNG_FOUND true PARENT_SCOPE)
set(PNG_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include" PARENT_SCOPE)
set(PNG_LIBRARY _libpng PARENT_SCOPE)
set(ZLIB_LIBRARY _zlib PARENT_SCOPE)
set(ExternConfig "${ExternConfig}
  add_library(_zlib STATIC IMPORTED)
  set_target_properties(_zlib PROPERTIES
    IMPORTED_LOCATION_RELEASE \"${CMAKE_INSTALL_PREFIX}/lib/zlibstatic.lib\"
    IMPORTED_LOCATION_RELWITHDEBINFO \"${CMAKE_INSTALL_PREFIX}/lib/zlibstatic.lib\"
    IMPORTED_LOCATION_DEBUG   \"${CMAKE_INSTALL_PREFIX}/lib/zlibstaticd.lib\"
  )
  add_library(_libpng STATIC IMPORTED)
  set_target_properties(_libpng PROPERTIES
    IMPORTED_LOCATION_RELEASE \"${CMAKE_INSTALL_PREFIX}/lib/libpng16_static.lib\"
    IMPORTED_LOCATION_RELWITHDEBINFO \"${CMAKE_INSTALL_PREFIX}/lib/libpng16_static.lib\"
    IMPORTED_LOCATION_DEBUG   \"${CMAKE_INSTALL_PREFIX}/lib/libpng16_staticd.lib\"
  )")
endif()

if( BUILD_EXTERN_LIBJPEG )

#########################################################
# libjpg
#########################################################

ExternalProject_Add( __libjpeg
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libjpeg"
  GIT_REPOSITORY https://github.com/LuaDist/libjpeg.git
  GIT_TAG bc8f8be222287fec977ec3f47a5cb065cceb2ee9
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
              -DBUILD_SHARED_LIBS=false
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
              -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
              -DCMAKE_C_FLAGS_MINSIZEREL=${CMAKE_C_FLAGS_MINSIZEREL}
              -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
)

add_library(_libjpeg STATIC IMPORTED GLOBAL)
add_dependencies(_libjpeg __libjpeg)
set_target_properties(_libjpeg PROPERTIES
    IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/jpeg.lib
)

set(JPEG_FOUND true PARENT_SCOPE)
set(JPEG_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include" PARENT_SCOPE)
set(JPEG_LIBRARY _libjpeg PARENT_SCOPE)
set(ExternConfig "${ExternConfig}
  add_library(_libjpeg STATIC IMPORTED)
  set_target_properties(_libjpeg PROPERTIES
    IMPORTED_LOCATION \"${CMAKE_INSTALL_PREFIX}/lib/jpeg.lib\"
  )")
endif()

set(ExternConfig "${ExternConfig}" PARENT_SCOPE)
