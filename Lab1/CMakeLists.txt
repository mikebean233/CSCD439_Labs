cmake_minimum_required(VERSION 3.5)
project(Lab1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES lab1.cu)
add_executable(Lab1 ${SOURCE_FILES})