
find_path(Lz4_INCLUDE_DIRS
    NAMES lz4frame.h
    PATHS 
        /opt/local/include
        /usr/local/include
        /usr/include
    )

find_library(Lz4_LIBRARIES
    NAMES lz4
    PATHS 
        /usr/local/lib
        /opt/local/lib
        /user/local/lib
        /usr/lib
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Lz4 REQUIRED_VARS Lz4_LIBRARIES Lz4_INCLUDE_DIRS)

mark_as_advanced(
    Lz4_INCLUDE_DIRS
    Lz4_LIBRARIES
    Lz4_FOUND
)
