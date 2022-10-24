#include <pybind11/pybind11.h>
#include <wrapper.h>

PYBIND11_MODULE(zebra, m) {
    m.doc() = R"pbdoc(
        C++ implementation of postprocessing operators on numpy chunks.
    )pbdoc";

    m.def("brn", &brn_np, R"pbdoc(
        Computes the bulk richardson number.
    )pbdoc");
    m.def("thetav", &thetav_np, R"pbdoc(
        Computes the virtual potential temperature for the bulk richardson number.
    )pbdoc");
}
