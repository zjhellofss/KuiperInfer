find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(../test DIR_TEST)
aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/runtime DIR_PARSER)
aux_source_directory(../source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(../source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(../source/layer/monocular DIR_MONOCULAR_LAYER)
aux_source_directory(../source/shape DIR_SHAPE)
aux_source_directory(../source/parser DIR_PARSER)

set(link_lib glog::glog pthread GTest::gtest)
set(link_math_lib ${ARMADILLO_LIBRARIES} blas lapack)

add_executable(test_kuiper ${DIR_TEST} ${DIR_DATA} ${DIR_PARSER} ${DIR_BINOCULAR_LAYER} ${DIR_MONOCULAR_LAYER} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER} ${DIR_SHAPE} test_hardswish.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp -march=native")

target_link_libraries(test_kuiper ${link_lib} ${link_math_lib})
target_link_libraries(test_kuiper OpenMP::OpenMP_CXX)

target_include_directories(bench_kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(bench_kuiper PUBLIC ${Armadillo_INCLUDE_DIR})


