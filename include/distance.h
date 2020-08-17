#ifndef CDIM_H
#define CDIM_H

#include <math.h>
#include <stdlib.h>

/** Hellinger distance for stochastic vectors.
 */
int
hellinger (const double *pva,
           const double *pvb,
           const size_t  len,
                 double *dist);

#endif  /* CDIM_H */
