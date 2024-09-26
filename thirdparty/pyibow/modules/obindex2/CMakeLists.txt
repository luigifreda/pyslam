cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(obindex2 LANGUAGES CXX)

# Catkin dependencies
find_package(catkin REQUIRED COMPONENTS roscpp)

# Defining the package
catkin_package(
    INCLUDE_DIRS lib/include
    LIBRARIES obindex2
    CATKIN_DEPENDS roscpp
    DEPENDS
)

###########
## Build ##
###########

# Compiling the library
add_subdirectory(lib)