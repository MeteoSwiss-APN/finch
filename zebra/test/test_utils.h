#include <concepts>

/**
 * Concept for specifying numeric typenames.
 */
template<typename T>
concept numeric = std::integral<T> or std::floating_point<T>;

/**
 * Checks whether the two given arrays are approximately equal.
 * 
 * @param absDelta The maximum absolute error allowed.
 * @param relDelta The maximum relative error allowed.
 */
template<typename T> requires numeric<T>
bool approx_equal(T *a, T* b, int size, T absDelta = 0, T relDelta = 0);