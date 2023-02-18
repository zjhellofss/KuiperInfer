cmake_minimum_required(VERSION 3.16)
project(kuiper_infer)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)
if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # 假如编译x64，那么SSE2永远可用
    # 假如编译x86，那么SSE2可用的话，就开启SSE2
    if (CMAKE_SIZEOF_VOID_P EQUAL 4)
        # check if SSE2 is available
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:SSE2" HAS_SSE2)
    else ()
        set(HAS_SSE2 ON)
    endif ()
    if (HAS_SSE2)
        # set __SSE2__ and __XOP__ macros
        add_definitions(-D__SSE2__ -D__XOP__)
    endif ()
endif ()

option(BUILD_DEMO "BUILD THE DEMO PROJECT")
set(BUILD_DEMO ON)

if (BUILD_DEMO)
    MESSAGE(STATUS "BUILD DEMO PROJECT")
    add_subdirectory(demos)
endif ()

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

aux_source_directory(./source/data DIR_DATA)
aux_source_directory(./source/runtime DIR_PARSER)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(./source/parser DIR_PARSER)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(link_lib glog::glog)
IF (!WIN32)
    set(link_lib ${link_lib} pthread)
ENDIF ()

add_library(kuiper SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER} source/layer/details/platform.hpp)
target_link_libraries(kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

target_include_directories(kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${Armadillo_INCLUDE_DIR})

# mathfun library defines
add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)

# 本项目的开发者请使用set(DEVELOPMENT ON)或者在cmake中添加-DDEVELOPMENT=ON将选项打开
option(DEVELOPMENT ON)
if (${DEVELOPMENT})
    message(STATUS "DEVELOPMENT MODE ON")
    enable_testing()
    add_subdirectory(bench)
    add_subdirectory(test)
endif ()