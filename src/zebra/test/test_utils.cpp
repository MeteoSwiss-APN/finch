#include <test_utils.h>

template<typename T> requires numeric<T>
bool approx_equal(T *a, T* b, int size, T absDelta, T relDelta) {
    for(int i = 0; i < size; i++) {
        if(abs(a[i] - b[i]) > absDelta + relDelta*abs(b[i])) {
            return false;
        }
    }
    return true;
}
