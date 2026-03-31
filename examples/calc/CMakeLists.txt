cmake_minimum_required(VERSION 3.20)
project(calc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE CALC_SOURCES src/*.cpp)

add_executable(calc ${CALC_SOURCES})
target_include_directories(calc PRIVATE src)

# -- static library of non-main sources (for tests to link against) ----------
set(LIB_SOURCES ${CALC_SOURCES})
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

if(LIB_SOURCES)
    add_library(calc_lib STATIC ${LIB_SOURCES})
    target_include_directories(calc_lib PUBLIC src)
endif()

# -- tests via doctest -------------------------------------------------------
enable_testing()

file(GLOB_RECURSE TEST_SOURCES tests/*_test.cpp)
foreach(TEST_SRC ${TEST_SOURCES})
    file(RELATIVE_PATH TEST_REL "${CMAKE_SOURCE_DIR}/tests" "${TEST_SRC}")
    string(REPLACE "/" "_" TEST_NAME "${TEST_REL}")
    string(REPLACE ".cpp" "" TEST_NAME "${TEST_NAME}")

    add_executable(${TEST_NAME} ${TEST_SRC})
    target_include_directories(${TEST_NAME} PRIVATE src tests)
    if(LIB_SOURCES)
        target_link_libraries(${TEST_NAME} PRIVATE calc_lib)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
