find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)

aux_source_directory(../bench DIR_BENCH)

set(link_lib glog::glog pthread benchmark::benchmark benchmark::benchmark_main)
set(link_math_lib ${ARMADILLO_LIBRARIES} blas lapack)

add_executable(bench_kuiper ${DIR_BENCH} )
target_link_directories(bench_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")

target_link_directories(bench_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(bench_kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)
target_link_libraries(bench_kuiper kuiper)

target_include_directories(bench_kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(bench_kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(bench_kuiper PUBLIC ${Armadillo_INCLUDE_DIR})

