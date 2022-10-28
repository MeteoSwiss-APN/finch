#include <brn.h>
#include <cmath>
#include <immintrin.h>
#include <vectorclass.h>
#include <vectormath_exp.h>
#include <util.h>
#include <omp.h>

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

    double *tv_sum = malloc_d(o);

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
                (tv_sum[j]*(u[ij]*u[ij] + v[ij]*v[ij]));
        }
        out[i+o-1] = 0;
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

    double *tv_sum = malloc_d(o);

    Vec8d nat(0, 1, 2, 3, 4, 5, 6, 7);

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
            Vec8d outij = PC_G * (hhlij - hsurfhi) * (oij - last_tv) * (o-j-nat) / 
                (tvj*(uij*uij + vij*vij));
            outij.store(&out[ij]);
        }
    }
    free(tv_sum);
}

void thetav_vec_par(const double *p, const double *t, const double *qv, double *out, int m, int n, int o) {

    const double pc_rvd = PC_R_V / PC_R_D;
    const double pc_rdocp = PC_R_D / PC_CP_D;
    const double pc_rvd_o = pc_rvd - 1.0;

    #pragma omp parallel for firstprivate(pc_rvd, pc_rdocp, pc_rvd_o)
    for(int i = 0; i < m*n*o; i+=8) {
        Vec8d ti, pi, qvi;
        ti.load(&t[i]);
        pi.load(&p[i]);
        qvi.load(&qv[i]);
        Vec8d outi = pow(P0 / pi,pc_rdocp) * ti * (1 + (pc_rvd_o * qvi / (1-qvi)));
        outi.store(&out[i]);
    }
}

void brn_vec_par(
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

    Vec8d nat(0, 1, 2, 3, 4, 5, 6, 7);

    double *tv_sum_full = malloc_d(o*omp_get_max_threads());

    // iterate over x,y locations
    #pragma omp parallel for firstprivate(tv_sum_full, nat)
    for(int hi = 0; hi < m*n; hi++) {
        double *tv_sum = &tv_sum_full[o*omp_get_thread_num()];
        int i = hi*o;
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
            Vec8d outij = PC_G * (hhlij - hsurfhi) * (oij - last_tv) * (o-j-nat) / 
                (tvj*(uij*uij + vij*vij));
            outij.store(&out[ij]);
        }
    }

    free(tv_sum_full);
}

void thetav(const double *p, const double *t, const double *qv, double *out, int m, int n, int o) {
    thetav_vec_par(p, t, qv, out, m, n, o);
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
    brn_vec_par(p, t, qv, u, v, hhl, hsurf, out, m, n, o);
}