#include <gtest/gtest.h>
#include <brn.h>
#include "test_utils.h"


TEST(BrnTests, MonoInput) {
    const int m = 1, n = 1, o = 80;
    double *x = (double *) std::malloc(sizeof(double) * n*m*o);
    for(int i = 0; i < n*m*o; i++) {
        x[i] = 2;
    }
    double *out = (double *) std::malloc(sizeof(double) * n*m*o);
    brn(x, x, x, x, x, x, x, out, m, n, o);
    free(x);
}