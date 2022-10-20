#ifndef BRN
#define BRN

void thetav(const double *p, const double *t, const double *qv, double *out, int m, int n, int o);
void brn(const double *p, const double *t, const double *qv, const double *u, const double *v, const double *hhl, const double *hsurf, double *out, int m, int n, int o);

#endif