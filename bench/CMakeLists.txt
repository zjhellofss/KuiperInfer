find_package(benchmark REQUIRED)
aux_source_directory(../bench DIR_BENCH)

set(link_lib benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_kuiper ${DIR_BENCH} ../demos/llama2/llama_chat.cpp)
target_link_directories(bench_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif()

target_link_directories(bench_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(bench_kuiper ${link_lib} OpenMP::OpenMP_CXX)
target_link_libraries(bench_kuiper kuiper)

if (MSVC)
    # find kuiper dll
    add_custom_command(TARGET bench_kuiper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:kuiper>/kuiper.dll"
            $<TARGET_FILE_DIR:bench_kuiper>)
endif()

target_include_directories(bench_kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(bench_kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(bench_kuiper PUBLIC ${Armadillo_INCLUDE_DIR})
