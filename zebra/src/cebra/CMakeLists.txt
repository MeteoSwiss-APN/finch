set(CMAKE_CXX_FLAGS "-march=native -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(FetchContent)

FetchContent_Declare(
  vectorclass
  GIT_REPOSITORY https://github.com/vectorclass/version2.git
  GIT_TAG v2.02.00
)
FetchContent_MakeAvailable(vectorclass)

find_package(OpenMP REQUIRED)

add_library(cebra
  brn.cpp
  config.cpp
)

target_include_directories(cebra
  PUBLIC include
  ${vectorclass_SOURCE_DIR}
)

target_link_libraries(cebra
  PUBLIC OpenMP::OpenMP_CXX
)

install(
  TARGETS cebra
  LIBRARY DESTINATION cebra
)