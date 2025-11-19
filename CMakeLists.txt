cmake_minimum_required(VERSION 3.14)
project(onnx_inference_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Добавляем директорию libs для httplib.h и json.hpp
include_directories(${CMAKE_SOURCE_DIR}/libs)

# ONNX Runtime
if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must be defined")
endif()
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# SentencePiece (установлен в /usr/local)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Главный исполняемый файл
add_executable(server src/main.cpp)
target_link_libraries(server PRIVATE Threads::Threads onnxruntime sentencepiece)
find_package(Threads REQUIRED)