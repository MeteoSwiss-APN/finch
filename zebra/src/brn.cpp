#include "brn.h"
#include <cmath>
#include <immintrin.h>
#include <vectorclass.h>
#include <vectormath_exp.h>

#define PC_G 9.80665
#define PC_R_D 287.05
#define PC_R_V 461.51
#define PC_CP_D 1005.0
#define P0 1.0e5

void thetav_naive(const double *p, const double *t, const double *qv, double *out, int m, int n, int o) {

    const double pc_rvd = PC_R_V / PC_R_D;
    const double pc_rdocp = PC_R_D / PC_CP_D;
    const double pc_rvd_o = pc_rvd - 1.0;

    for(int i = 0; i < m*n*o; i++) {
        out[i] = pow(P0 / p[i], pc_rdocp) * t[i] * (1. + (pc_rvd_o * qv[i] / (1.-qv[i])));
    }
}

void brn_naive(
    const double *p, 
    const double *t, 
    const double *qv, 
    const double *u, 
    const double *v, 
    const double *hhl, 
    const double *hsurf, 
    double *out, 
    int m, int n, int o) {
    
    thetav(p, t, qv, out, m, n, o);

    double *tv_sum = (double *) malloc(sizeof(double) * o);

    // iterate over x,y locations
    for(int i = 0, hi = 0; i < m*n*o; i += o, hi++) {
        // compute thetav cumsum for current x,y location
        double prev = 0;
        for(int j = 0; j < o; j++) {
            double csj = out[i+o-1-j] + prev;
            tv_sum[j] = csj;
            prev = csj;
        }
        // compute brn
        double last_tv = out[i+o-1];
        for(int j = 0; j < o-1; j++) {
            int ij = i+j;
            out[ij] = PC_G * (hhl[ij] - hsurf[hi]) * (out[ij] - last_tv) * (o-j) / 
                (tv_sum[j]*(u[ij]*u[ij] - v[ij]*v[ij]));
        }
    }
    free(tv_sum);
}

void thetav_vec(const double *p, const double *t, const double *qv, double *out, int m, int n, int o) {

    const double pc_rvd = PC_R_V / PC_R_D;
    const double pc_rdocp = PC_R_D / PC_CP_D;
    const double pc_rvd_o = pc_rvd - 1.0;

    for(int i = 0; i < m*n*o; i+=8) {
        Vec8d ti, pi, qvi;
        ti.load(&t[i]);
        pi.load(&p[i]);
        qvi.load(&qv[i]);
        Vec8d outi = pow(P0 / pi,pc_rdocp) * ti * (1 + (pc_rvd_o * qvi / (1-qvi)));
        outi.store(&out[i]);
    }
}

void brn_vec(
    const double *p, 
    const double *t, 
    const double *qv, 
    const double *u, 
    const double *v, 
    const double *hhl, 
    const double *hsurf, 
    double *out, 
    int m, int n, int o) {
    
    thetav(p, t, qv, out, m, n, o);

    double *tv_sum = (double *) malloc(sizeof(double) * o);

    // iterate over x,y locations
    for(int i = 0, hi = 0; i < m*n*o; i += o, hi++) {
        // compute thetav cumsum for current x,y location
        double prev = 0;
        for(int j = 0; j < o; j++) {
            double csj = out[i+o-1-j] + prev;
            tv_sum[j] = csj;
            prev = csj;
        }
        // compute brn
        double last_tv = out[i+o-1];
        double hsurfhi = hsurf[hi];
        for(int j = 0; j < o; j+=8) {
            int ij = i+j;
            Vec8d oij, hhlij, tvj, uij, vij;
            oij.load(&out[ij]);
            hhlij.load(&hhl[ij]);
            tvj.load(&tv_sum[j]);
            uij.load(&u[ij]);
            vij.load(&v[ij]);
            Vec8d outij = PC_G * (hhlij - hsurfhi) * (oij - last_tv) * (o-j) / 
                (tvj*(uij*uij - vij*vij));
            outij.store(&out[ij]);
        }
    }
    free(tv_sum);
}

void thetav(const double *p, const double *t, const double *qv, double *out, int m, int n, int o) {
    thetav_naive(p, t, qv, out, m, n, o);
}

void brn(
    const double *p, 
    const double *t, 
    const double *qv, 
    const double *u, 
    const double *v, 
    const double *hhl, 
    const double *hsurf, 
    double *out, 
    int m, int n, int o) {
    brn_naive(p, t, qv, u, v, hhl, hsurf, out, m, n, o);
}