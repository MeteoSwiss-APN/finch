find_package(pybind11 REQUIRED)

pybind11_add_module(zebra
    bindings.cpp
    wrapper.cpp
)

target_include_directories(zebra
    PUBLIC include
)

target_link_libraries(zebra
    PRIVATE cebra
)

install(
    TARGETS zebra
    DESTINATION zebra
)
