#ifndef CDIM_H
#define CDIM_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/** Condensed distance matrix of delay embedding
 */
void
delay_embedding_dists (const double *inp,
                       const size_t  n_vectors,
                       const size_t  delay,
                       const size_t  m_dim,
                             double *dists);

void
comsar_fractal_embedding (const double *x,
                          const size_t  N_max,
                          const size_t  m_dim,
                          const size_t  delay,
                          const size_t  n_dist,
                                double *dist,
                                double *d_min,
                                double *d_max);

void
comsar_fractal_correlation_sum (const size_t n_radius,
                                const double *radius,
                                const size_t n_dist, const double *dist,
                                const size_t N_max, double *Cr);

int
comsar_fractal_csum (const double *sig,
                     const size_t  n_sig,
                     const double *radius,
                     const size_t  n_radius,
                     const size_t  m_dim,
                     const size_t  delay,
                           double *Cr);

int
comsar_fractal_cdim (const double *x,
                     const size_t  N,
                     const size_t  n_radius,
                     const size_t  m_dim,
                     const size_t  delay,
                           double *Cr);

double
corr_dim_bader (const short  *snd,
                const size_t  delay,
                const size_t  m_dim,
                const size_t  n_bins,
                const size_t  slope_points);

#endif  /* CDIM_H */
