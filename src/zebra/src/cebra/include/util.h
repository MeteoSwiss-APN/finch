#ifndef CEBRA_UTIL_H
#define CEBRA_UTIL_H

#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <sys/time.h>
#include <cmath>

#define malloc_t(s, t) ((t *) std::malloc(sizeof(t) * (s)))
#define malloc_d(s) malloc_t(s, double)

#endif
