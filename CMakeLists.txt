cmake_minimum_required(VERSION 3.23)
project(kuiper_infer)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)

set(USE_CUDA "Support Cuda Backend" ON)
set(BUILD_DEMO "Build The Demo Project" ON)

if (BUILD_DEMO)
    MESSAGE(STATUS "Build Demo Project")
    add_subdirectory(demos)
endif ()

if (USE_CUDA)
    MESSAGE(STATUS "Support Cuda Backend")
    add_definitions(-DUSE_CUDA)
endif ()

enable_testing()
add_subdirectory(bench)
add_subdirectory(test)
