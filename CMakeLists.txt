cmake_minimum_required(VERSION 4.2)
project(neuralNetworkFS)

set(EXEC program)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(BASEPATH "${CMAKE_SOURCE_DIR}")

add_executable(${EXEC} src/main.cpp)
