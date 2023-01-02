cmake_minimum_required(VERSION 3.9)#Cmake最低版本
#project(demo LANGUAGES CXX CUDA) #这个写法也是可以的
project(demo)
enable_language(CUDA)#激活CUDA语言支持，使用第一个写法时要进行注释

find_package(CUDA)

#设置语言标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)   

#添加项目自身的库和依赖
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)#设置编译后的程序存放位置
aux_source_directory(src SRC_LIST)#添加src目录下的所有源文件
set (SRC_LIST
    src/main.cpp
    src/cuda/im2col.cu
    src/cuda/gemm.cu
    )
include_directories(include)#添加头文件路径

include_directories("/usr/local/cuda-11.7/targets/x86_64-linux/include")#添加头文件路径

#添加OpenCV库（这个库比较常用，所以写个注释在这里）
#find_package(OpenCV 4 REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#message(STATUS "include:${OpenCV_INCLUDE_DIRS}")
#link_libraries(${OpenCV_LIBS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

#编译列表
add_executable(${PROJECT_NAME} ${SRC_LIST})
