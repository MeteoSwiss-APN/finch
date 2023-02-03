#include <brn.h>
#include <util.h>
#include <mock.h>
#include <config.h>
#include <chrono>

void profile_brn(int max_threads = 10) {
    const int m = 1000, n = 100, o = 80;
    double *x = malloc_d(m*n*o*7 + m*n);
    double *p = x,
        *t = &x[m*n*o],
        *qv = &x[m*n*o*2],
        *u = &x[m*n*o*3],
        *v = &x[m*n*o*4],
        *hhl = &x[m*n*o*5],
        *hsurf = &x[m*n*o*7],
        *out = &x[m*n*o*6];
    brn_mock(p, t, qv, u, v, hhl, hsurf, out, m, n, o);

    double *times = new double[max_threads];

    for(int i = 0; i < max_threads; i++) {
        int threads = i+1;
        set_threads(threads);
        auto start = std::chrono::steady_clock::now();
        brn(p, t, qv, u, v, hhl, hsurf, out, m, n, o);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;
        double seconds = duration.count();
        std::cout << "Time for " << threads << " threads: " << seconds << "s" << std::endl;
        times[i] = seconds;
    }

    // calculate scaling factor
    // this is just simple linear regression on the log-log-plot.
    double *logx = new double[max_threads];
    double *logy = new double[max_threads];
    double xm = 0, ym = 0;
    for(int i = 0; i < max_threads; i++) {
        double lx = log(i+1);
        double ly = log(times[i]);
        logx[i] = lx;
        logy[i] = ly;
        xm += lx;
        ym += ly;
    }
    xm /= max_threads;
    ym /= max_threads;
    double a1 = 0, a2 = 0;
    for(int i = 0; i < max_threads; i++) {
        double xd = (logx[i] - xm);
        a1 += xd * (logy[i] - ym);
        a2 += xd*xd;
    }
    double a = a1 / a2;
    double b = ym - a*xm;
    std::cout << "Fitted scaling rate: " << -a << " (1 is perfect scaling)" << std::endl;
    std::cout << "Fitted scaling factor: " << b << " (1 is perfect scaling)" << std::endl;
    double mean_rel_err = 0;
    for(int i = 0; i < max_threads; i++) {
        double pred = a*logx[i] + b;
        mean_rel_err += abs((logy[i] - pred) / logy[i]);
    }
    mean_rel_err /= max_threads;
    std::cout << "Mean relative error: " << mean_rel_err << " (0 is ideal)" << std::endl;
}

int main() {
    profile_brn();
}
