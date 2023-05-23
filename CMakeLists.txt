cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)

project(kuiper_infer CXX CUDA)
#set(CMAKE_CXX_STANDARD 17)


set(BUILD_DEMO OFF)
set(USE_GPU ON)

#使用GPU
add_definitions(-DGPU_ONLY)

# add_definitions(-)

#CUDA需要的头文件
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})





option(BUILD_DEMO "BUILD THE DEMO PROJECT")

#option(USE_GPU "USE GPU" ON)


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
    # Force LLVM OpenMP on MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm /openmp:experimental")
endif ()


#if (BUILD_DEMO)
#    MESSAGE(STATUS "BUILD DEMO PROJECT")
#    add_subdirectory(demos)
#endif ()

# 定义变量来表示情况A是否成立
set(IS_CASE_A TRUE)

# 添加其他文件到库中
set(SOURCES other.cpp)

# 根据情况A添加对应的文件
#if (IS_CASE_A)
#    set(SOURCES ${SOURCES} a_gpu.cpp)
#else ()
#    set(SOURCES ${SOURCES} a.cpp)
#endif ()
#
## 创建库并添加源文件
#add_library(my_library ${SOURCES})



# find_package(benchmark REQUIRED)
# find_package(OpenMP REQUIRED)
# find_package(Armadillo REQUIRED)
 find_package(glog REQUIRED)
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)

include_directories(./include)

if (USE_GPU)



    MESSAGE(STATUS "BUILD  PROJECT  WITH GPU")
    # aux_source_directory(./source/data DIR_DATA)
    aux_source_directory(./source/runtime DIR_PARSER)
    aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)

#    aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)

    file(GLOB LAYER_DETAILS "./source/layer/details/*.hpp" "./source/layer/details/*.cu")
    # add_library(DIR_BINOCULAR_LAYER ${LAYER_DETAILS_SOURCES})
   

    file(GLOB DIR_DATA "./source/data/tensor.cpp" "./source/data/tensor_utils.cu")
    # add_library(DIR_DATA ${DIR_DATA_SOURCES})

    aux_source_directory(./source/parser DIR_PARSER)
    aux_source_directory(./source/utils DIR_UTILS)


    add_library(kuiper SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_UTILS} ${DIR_ABSTRACT_LAYER} ${DIR_LAYER_DETAILS} ${DIR_PARSER})

#    add_library(kuiper SHARED ${DIR_DATA})
#    target_link_libraries(kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)
    target_include_directories(kuiper PUBLIC ${glog_INCLUDE_DIR})


else ()
    aux_source_directory(./source/data DIR_DATA)
    aux_source_directory(./source/runtime DIR_PARSER)
    aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
    aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
    aux_source_directory(./source/parser DIR_PARSER)
    aux_source_directory(./source/utils DIR_UTILS)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set(link_lib glog::glog)
    IF (!WIN32)
        set(link_lib ${link_lib} pthread)
    ENDIF ()

    set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

    add_library(kuiper SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_UTILS} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER})
    target_link_libraries(kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

    target_include_directories(kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
    target_include_directories(kuiper PUBLIC ${glog_INCLUDE_DIR})
    target_include_directories(kuiper PUBLIC ${GTest_INCLUDE_DIR})
    target_include_directories(kuiper PUBLIC ${Armadillo_INCLUDE_DIR})
endif ()


# find_package(benchmark REQUIRED)
# find_package(OpenMP REQUIRED)
# find_package(Armadillo REQUIRED)
# find_package(glog REQUIRED)
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)


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

