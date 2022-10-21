#ifndef UTIL
#define UTIL

#include <stdlib.h>

#define malloc_t(s, t) ((t *) std::malloc(sizeof(t) * (s)))
#define malloc_d(s) malloc_t(s, double)

#endif