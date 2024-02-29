cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
endif ()
set(CMAKE_BUILD_TYPE Release)
add_executable(llama_chat llama_chat.cpp main.cpp)
target_link_directories(llama_chat PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_chat ${OpenCV_LIBS} kuiper)

