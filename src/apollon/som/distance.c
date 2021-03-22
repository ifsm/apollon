#if !defined(__clang__) && defined(__GNUC__) && defined(__GNUC_MINOR__)
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
#pragma GCC optimize("tree-vectorize")
#pragma GCC optimize("unsafe-math-optimizations")
#pragma GCC optimize("unroll-loops")
#pragma GCC diagnostic warning "-Wall"
#endif
#endif
#include <stdio.h>
#include "distance.h"


int
hellinger (const double *pva,
           const double *pvb,
           const size_t  n_elements,
                 double *dist)
{
    for (size_t i = 0; i < n_elements; i++)
    {
        double diff = sqrt (pva[i]) - sqrt (pvb[i]);
        *dist += diff * diff;
    }
    *dist = sqrt (*dist / 2);
    return 1;
}
