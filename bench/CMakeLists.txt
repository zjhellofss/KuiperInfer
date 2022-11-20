find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(./include/math)

aux_source_directory(../bench DIR_BENCH)
aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/parser DIR_PARSER)
aux_source_directory(../source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(../source/layer/binocular DIR_BINOCULAR_LAYER)
aux_source_directory(../source/layer/monocular DIR_MONOCULAR_LAYER)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
find_package(OpenMP REQUIRED)

set(link_lib glog pthread gflags gtest)
set(link_math_lib armadillo blas lapack)
add_executable(bench_kuiper ${DIR_BENCH} ${DIR_DATA} ${DIR_BINOCULAR_LAYER} ${DIR_MONOCULAR_LAYER} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER})
target_link_libraries(bench_kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX benchmark::benchmark benchmark::benchmark_main)


