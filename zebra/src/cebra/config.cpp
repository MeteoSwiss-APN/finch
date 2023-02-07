#include <config.h>
#include <omp.h>

void set_threads(int n) {
    omp_set_num_threads(n);
}

int get_threads() {
    return omp_get_max_threads();
}
