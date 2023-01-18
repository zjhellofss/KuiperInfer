cmake_minimum_required(VERSION 3.16)
project(yolo_test)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

aux_source_directory(../demos DIR_DEMOS)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /openmp:llvm")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif()

add_executable(yolo_test ${DIR_DEMOS} image_util.hpp image_util.cpp)
target_include_directories(yolo_test PUBLIC ../include)
target_link_directories(yolo_test PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(yolo_test ${OpenCV_LIBS} kuiper)

if (MSVC)
    add_custom_command(TARGET yolo_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:kuiper>/kuiper.dll"
            $<TARGET_FILE_DIR:yolo_test>)
endif()