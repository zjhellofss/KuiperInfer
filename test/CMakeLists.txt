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
  hello_test
  ./test_data/test_tensor.cpp
)
target_link_libraries(
  hello_test
  GTest::gtest_main
  kuiper
  glog::glog 
)

include(GoogleTest)
gtest_discover_tests(hello_test)

# add_test(multiply_gtests multiply_test)

# add_executable(example   )
# # target_link_directories(test_kuiper PUBLIC ${PROJECT_SOURCE_DIR}/lib)
# target_link_libraries(example  GTest::GTest)
# target_link_libraries(example )
