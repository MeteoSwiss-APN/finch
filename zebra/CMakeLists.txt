cmake_minimum_required(VERSION 3.14)
project(zebra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(FetchContent)

FetchContent_Declare(
  vectorclass
  GIT_REPOSITORY https://github.com/vectorclass/version2.git
  GIT_TAG v2.02.00
)
FetchContent_MakeAvailable(vectorclass)

find_package(pybind11 REQUIRED)

include_directories(
  include
  ${vectorclass_SOURCE_DIR}
)

pybind11_add_module(zebra 
    src/bindings.cpp
    src/wrapper.cpp
    src/brn.cpp
)

add_library(cebra
  src/brn.cpp
)

# tests

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(
    zebra_test
    test/brn_test.cpp
    test/test_utils.cpp
)

include(GoogleTest)
gtest_discover_tests(zebra_test)

enable_testing()

target_link_libraries(
    zebra_test
    GTest::gtest_main
    cebra
)