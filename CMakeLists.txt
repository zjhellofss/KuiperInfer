cmake_minimum_required(VERSION 3.23)
project(kuiper_course)

set(CMAKE_CXX_STANDARD 17)

### gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

### armadillo
find_package(glog REQUIRED)

### gflag
find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

### arma
include_directories(./include/math)

### json
include_directories(./include/json)

include_directories(./include)

enable_testing()
add_subdirectory(test)

set(link_lib glog pthread gflags gtest)
set(link_math_lib armadillo blas lapack)

add_executable(kuiper_course main.cpp)
target_link_libraries(kuiper_course ${link_lib} ${link_math_lib})


