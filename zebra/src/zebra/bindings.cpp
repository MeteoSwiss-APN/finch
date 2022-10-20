#include <wrapper.h>

PYBIND11_MODULE(zebra, m) {
    m.doc() = "C++ implementation of postprocessing operators on numpy chunks.";

    m.def("brn", &brn_np, "Computes the bulk richardson number.");
    m.def("thetav", &thetav_np, "Computes the virtual potential temperature for the bulk richardson number.");
}
