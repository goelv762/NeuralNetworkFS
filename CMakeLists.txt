cmake_minimum_required(VERSION 4.2)
project(neuralNetworkFS)

set(EXEC program)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(BASEPATH "${CMAKE_SOURCE_DIR}")
file(GLOB_RECURSE MATH src/math/*.cpp)
file(GLOB_RECURSE NETWORK src/network/*.cpp)
file(GLOB_RECURSE DATA src/data/*.cpp)

set(VENV_PYTHON ${CMAKE_SOURCE_DIR}/venv/bin/python)
set(SCRIPT ${CMAKE_SOURCE_DIR}/data/spiral.py)

file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CMAKE_BINARY_DIR}")

add_custom_target(genData ALL
    COMMAND ${VENV_PYTHON} ${SCRIPT}
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_library(math SHARED ${MATH})
set_target_properties(math PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")

add_library(network SHARED ${NETWORK})
set_target_properties(network PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")

add_library(data SHARED ${DATA})
set_target_properties(data PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++")

add_executable(${EXEC} src/main.cpp)
add_dependencies(${EXEC} genData)
target_link_libraries(${EXEC} math network data)
