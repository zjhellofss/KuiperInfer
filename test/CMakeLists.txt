find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(../include/math)
include_directories(../include/json)

aux_source_directory(../test DIR_TEST)
aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/layer DIR_LAYER)


set(link_lib glog pthread gflags gtest)
set(link_math_lib armadillo blas lapack)
add_executable(test_kuiper ${DIR_TEST} ${DIR_DATA} ${DIR_LAYER})
target_link_libraries(test_kuiper ${link_lib} ${link_math_lib})