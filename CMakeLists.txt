cmake_minimum_required(VERSION 3.23)
project(kuiper_infer)
set(CMAKE_CXX_STANDARD 17)
include_directories(./include)

enable_testing()
add_subdirectory(bench)
add_subdirectory(test)