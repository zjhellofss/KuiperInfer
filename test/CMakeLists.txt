find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(../test DIR_TEST)

set(link_lib glog::glog pthread GTest::gtest)
set(link_math_lib ${ARMADILLO_LIBRARIES} blas lapack)

add_executable(test_kuiper ${DIR_TEST})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp -march=native")

target_link_libraries(test_kuiper ${link_lib} ${link_math_lib})
target_link_libraries(test_kuiper OpenMP::OpenMP_CXX)

target_link_directories(test_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_kuiper kuiper)

target_include_directories(test_kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_kuiper PUBLIC ${Armadillo_INCLUDE_DIR})


