#include <gtest/gtest.h>
#include <brn.h>
#include <test_utils.h>
#include <util.h>
#include <mock.h>

TEST(ThetaVTests, Extensive) {
    const int m = 2, n = 2, o = 8;
    double *x = malloc_d(m*n*o*5);
    double *p = x, 
        *t = &x[m*n*o], 
        *qv = &x[m*n*o*2],
        *out_ref = &x[m*n*o*3],
        *out = &x[m*n*o*4];
    thetav_mock(p, t, qv, out_ref, m, n, o);
    thetav(p, t, qv, out, m, n, o);
    for(int i = 0; i < m*n*o; i++) {
        EXPECT_DOUBLE_EQ(out[i], out_ref[i]);
    }
    free(x);
}

TEST(BrnTests, ZeroOutput) {
    const int m = 1, n = 1, o = 8;
    double x[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7};
    double y[8] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    double out[8];
    brn(x, x, x, x, y, y, y, out, m, n, o);
    for(int i = 0; i < m*n*o; i++) {
        EXPECT_DOUBLE_EQ(out[i], 0) << "Output must be zero at index " << i << " but is " << out[i];
    }
}


TEST(BrnTests, Extensive) {
    const int m = 2, n = 2, o = 8;
    double *x = malloc_d(m*n*o*8 + m*n);
    double *p = x, 
        *t = &x[m*n*o], 
        *qv = &x[m*n*o*2], 
        *u = &x[m*n*o*3], 
        *v = &x[m*n*o*4], 
        *hhl = &x[m*n*o*5], 
        *hsurf = &x[m*n*o*8], 
        *out_ref = &x[m*n*o*6],
        *out = &x[m*n*o*7];
    brn_mock(p, t, qv, u, v, hhl, hsurf, out_ref, m, n, o);
    brn(p, t, qv, u, v, hhl, hsurf, out, m, n, o);
    for(int i = 0; i < m*n*o; i++) {
        EXPECT_DOUBLE_EQ(out[i], out_ref[i]) << "Output differs at index " << i;
    }
    free(x);
}