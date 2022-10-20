include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(zebra_test
    brn_test.cpp
    test_utils.cpp
)

target_include_directories(zebra_test
    PRIVATE include
)

target_link_libraries(zebra_test
    PRIVATE
        GTest::gtest_main
        cebra
)

include(GoogleTest)
gtest_discover_tests(zebra_test)