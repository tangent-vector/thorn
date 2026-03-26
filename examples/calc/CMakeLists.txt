cmake_minimum_required(VERSION 3.20)
project(calc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE CALC_SOURCES src/*.cpp)

add_executable(calc ${CALC_SOURCES})
target_include_directories(calc PRIVATE src)
