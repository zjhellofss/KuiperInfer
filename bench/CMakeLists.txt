find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)

aux_source_directory(../bench DIR_BENCH)
aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/runtime DIR_RUNTIME)
aux_source_directory(../source/parser DIR_PARSER)

aux_source_directory(../source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(../source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(../source/layer/monocular DIR_MONOCULAR_LAYER)

set(link_lib glog::glog pthread benchmark::benchmark benchmark::benchmark_main)
set(link_math_lib ${ARMADILLO_LIBRARIES} blas lapack)

add_executable(bench_kuiper ${DIR_BENCH} ${DIR_DATA} ${DIR_BINOCULAR_LAYER} ${DIR_RUNTIME} ${DIR_MONOCULAR_LAYER} ${DIR_PARSER} ${DIR_ABSTRACT_LAYER})
target_link_libraries(bench_kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)
