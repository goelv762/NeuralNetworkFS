cmake_minimum_required(VERSION 4.2)
project(neuralNetworkFS)

set(EXEC program)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(BASEPATH "${CMAKE_SOURCE_DIR}")
file(GLOB_RECURSE MATH src/math/*.cpp)
file(GLOB_RECURSE NETWORK src/network/*.cpp)

add_library(math SHARED ${MATH})
set_target_properties(math PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")

add_library(network SHARED ${NETWORK})
set_target_properties(network PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")

add_executable(${EXEC} src/main.cpp)
target_link_libraries(${EXEC} math network)
