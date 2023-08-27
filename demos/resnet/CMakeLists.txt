cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif ()

add_executable(resnet_test resnet_test.cpp ../image_util.hpp ../image_util.cpp)
target_link_directories(resnet_test PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(resnet_test ${OpenCV_LIBS} kuiper)

