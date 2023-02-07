#include <mock.h>

void thetav_mock(double *p, double *t, double *qv, double *out, int m, int n, int o) {
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

void brn_mock(double *p, double *t, double *qv, double *u, double *v, double *hhl, double *hsurf, double *out, int m, int n, int o) {
    double uc = 42, vc = 69, hhlc = 3, hsurfc = 1.5;
    thetav_mock(p, t, qv, out, m, n, o);
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
