cmake_minimum_required(VERSION 3.14)
project(onnx_inference_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/libs)

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must be defined")
endif()
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(server src/main.cpp)
target_link_libraries(server PRIVATE Threads::Threads onnxruntime sentencepiece)
find_package(Threads REQUIRED)