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

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)

aux_source_directory(./source/data DIR_DATA)
aux_source_directory(./source/runtime DIR_PARSER)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(./source/parser DIR_PARSER)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(link_lib glog::glog pthread)
set(link_math_lib ${ARMADILLO_LIBRARIES} blas lapack)

add_library(kuiper  ${DIR_DATA} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER} )
target_link_libraries(kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${Armadillo_INCLUDE_DIR})


enable_testing()
add_subdirectory(bench)
add_subdirectory(test)