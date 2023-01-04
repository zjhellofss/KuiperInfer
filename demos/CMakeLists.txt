cmake_minimum_required(VERSION 3.23)
project(yolo_test)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
aux_source_directory(../demos DIR_DEMOS)

add_executable(yolo_test ${DIR_DEMOS})
target_link_libraries(yolo_test ${OpenCV_LIBS})