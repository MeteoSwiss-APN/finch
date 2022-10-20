#ifndef WRAPPER
#define WRAPPER

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <brn.h>

namespace py = pybind11;
typedef py::array_t<double> pbarray;

void brn_np(pbarray p, pbarray t, pbarray qv, pbarray u, pbarray v, pbarray hhl, pbarray hsurf, pbarray out);
void thetav_np(pbarray p, pbarray t, pbarray qv, pbarray out);

#endif