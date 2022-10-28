#include <config.h>
#include <omp.h>

int thread_count = 0;

void set_threads(int n) {
    thread_count = n;
    omp_set_num_threads(n);
}

int get_threads() {
    return thread_count;
}