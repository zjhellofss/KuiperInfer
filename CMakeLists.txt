cmake_minimum_required(VERSION 3.23)
project(kuiper_infer)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)
### gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

### glog
find_package(glog REQUIRED)

### gflag
find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

enable_testing()
add_subdirectory(bench)
add_subdirectory(test)