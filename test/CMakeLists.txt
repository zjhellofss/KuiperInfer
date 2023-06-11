include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


find_package(glog REQUIRED)



enable_testing()

add_executable(
  tensor_test
  ./test_data/test_tensor.cpp
)
target_link_libraries(
  tensor_test
  GTest::gtest_main
  kuiper
  glog::glog 
)



add_executable(
  layer_test
  ./test_layer/test_layer.cpp
)
target_link_libraries(
  layer_test
  GTest::gtest_main
  kuiper
  glog::glog 
)

add_executable(
  maxpooling_test
  ./test_layer/test_maxpooling.cpp
)

target_link_libraries(
  maxpooling_test
  GTest::gtest_main
  kuiper
  glog::glog 
)




include(GoogleTest)

# add_test(multiply_gtests multiply_test)

# add_executable(example   )
# # target_link_directories(test_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
# target_link_libraries(example  GTest::GTest)
# target_link_libraries(example )
