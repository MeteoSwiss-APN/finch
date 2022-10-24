#include <gtest/gtest.h>
#include <brn.h>
#include <test_utils.h>
#include <util.h>

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

void fill_thetav_test_inout(double *p, double *t, double *qv, double *out, int m, int n, int o) {
    double pc = 1023.4, tc = 127.3, qvc = 0.3;
    double single_out = 593.9314166503111;
    double top_out = 589.2658124346763;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < o; k++) {
                int ii = i*n*o + j*o + k;
                p[ii] = pc;
                t[ii] = k < o-1 ? tc : tc-1;
                qv[ii] = qvc;
                out[ii] = k < o-1 ? single_out : top_out;
            }
        }
    }
}

void fill_brn_test_inout(double *p, double *t, double *qv, double *u, double *v, double *hhl, double *hsurf, double *out, int m, int n, int o) {
    double uc = 42, vc = 69, hhlc = 3, hsurfc = 1.5;
    fill_thetav_test_inout(p, t, qv, out, m, n, o);
    double tv = out[0], tv_top = out[o-1];
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < o; k++) {
                int ii = i*n*o + j*o + k;
                u[ii] = uc;
                v[ii] = vc;
                hhl[ii] = hhlc;
                double cs = tv_top + k*tv;
                double cur_tv = k < o-1 ? tv : tv_top;
                out[ii] = 9.80665 * (hhlc-hsurfc) * (cur_tv - tv_top) * (o-k) / (cs * (uc*uc + vc*vc));
            }
            hsurf[i*n+j] = hsurfc;
        }
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
    fill_brn_test_inout(p, t, qv, u, v, hhl, hsurf, out_ref, m, n, o);
    brn(p, t, qv, u, v, hhl, hsurf, out, m, n, o);
    for(int i = 0; i < m*n*o; i++) {
        EXPECT_DOUBLE_EQ(out[i], out_ref[i]) << "Output differs at index " << i;
    }
    free(x);
}