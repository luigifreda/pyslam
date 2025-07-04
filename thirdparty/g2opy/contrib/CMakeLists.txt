include_directories(${PROJECT_SOURCE_DIR})


add_library(contrib SHARED ${G2O_LIB_TYPE}
    sba_addons/types_stereo_sba.cpp
    estimate_propagator/SmoothEstimatePropagator.hpp 
    estimate_propagator/SmoothEstimatePropagator.cpp
)


set_target_properties(contrib PROPERTIES OUTPUT_NAME ${LIB_PREFIX}contrib)
target_link_libraries(contrib 
    core
    types_sba
)