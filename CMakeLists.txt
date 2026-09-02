cmake_minimum_required(VERSION 3.16)
project(learned_index_randomization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Fetch GoogleTest for correctness tests
include(FetchContent)

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Common interface library (includes rndbench/rng.hpp for seeding)
add_library(rndbench_common INTERFACE)
target_include_directories(rndbench_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(rndbench_common INTERFACE cxx_std_17)

# Learned index library
add_library(learned_index_lib INTERFACE)
target_include_directories(learned_index_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(learned_index_lib INTERFACE rndbench_common)

# Correctness tests
add_executable(learned_index_test tests/test_learned_index.cpp)
target_link_libraries(learned_index_test PRIVATE learned_index_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(learned_index_test)
