#include "cdim.h"

void
delay_embedding_dists (const double *inp,
                       const size_t  n_vectors,
                       const size_t  delay,
                       const size_t  m_dim,
                             double *dists)
{
    
    for (size_t i = 0; i < n_vectors - 1; i++)
    {
        for (size_t j = i + 1; j < n_vectors; j++)
        {
            size_t flat_idx = i * n_vectors + j - i*(i+1)/2 - i - 1;
            for (size_t m = 0; m < m_dim; m++)
            {
                dists[flat_idx] += pow (inp[i+m*delay] - inp[j+m*delay], 2);
            }
            dists[flat_idx] = sqrt (dists[flat_idx]);
        }
    }
}

void
comsar_fractal_embedding (const double *x,
                          const size_t  N_max,
                          const size_t  m_dim,
                          const size_t  delay,
                          const size_t  n_dist,
                                double *dist,
                                double *d_min,
                                double *d_max)
{
    /* embedding */
    for (size_t i = 0; i < m_dim; i++)
    {
        /* distance matrix */
        for (size_t j = 0; j < N_max-1; j++)
        {
            for (size_t k = j+1, idx = 0; k < N_max; k++)
            {
                idx = j * N_max + k - j * (j+1) / 2 - j - 1;
                dist[idx] += pow (x[i+j*delay] - x[i+k*delay], 2);
            }
        }   /* END distance matrix */
    }   /* END embedding */

    *d_min = sqrt (dist[0]);
    *d_max = sqrt (dist[0]);
    for (size_t i = 0; i < n_dist; i++)
    {
        dist[i] = sqrt (dist[i]);
        if (dist[i] < *d_min) *d_min = dist[i];
        if (dist[i] > *d_max) *d_max = dist[i];
    }
}


void
comsar_fractal_correlation_sum (const size_t  n_radius,
                                const double *radius,
                                const size_t  n_dist,
                                const double *dist,
                                const size_t  N_max,
                                      double *Cr)
{
    for (size_t i = 0; i < n_radius; i++)
    {
        Cr[i] = 0;
        for (size_t j = 0; j < n_dist; j++)
        {
            if (dist[j] < radius[i])
            {
                Cr[i] += 1.0;
            }
        }
        Cr[i] *= 2.0 / ((double) N_max * ((double) N_max - 1.0));
    }
}


int
comsar_fractal_csum (const double *sig,
                     const size_t  n_sig,
                     const double *radius,
                     const size_t  n_radius,
                     const size_t  m_dim,
                     const size_t  delay,
                           double *Cr)
{
    const size_t  N_max  = (n_sig - m_dim) / delay + 1;
    const size_t  n_dist = N_max * (N_max - 1) / 2;
          double *dist   = NULL;
          double  d_min  = 0;
          double  d_max  = 0;

    dist = calloc (n_dist, sizeof (double));
    if (dist == NULL)
    {
        fprintf (stderr, "Out of memory while allocating `dist` in correlation sum.");
        return -1;
    }

    comsar_fractal_embedding (sig, N_max, m_dim, delay, n_dist, dist, &d_min, &d_max);
    comsar_fractal_correlation_sum (n_radius, radius, n_dist, dist, N_max, Cr);

    free (dist);
    return 0;
}


int
comsar_fractal_cdim (const double *x,
                     const size_t  N,
                     const size_t  n_radius,
                     const size_t  m_dim,
                     const size_t  delay,
                           double *Cr)
{
    const size_t  N_max  = (N - m_dim) / delay + 1;
    const size_t  n_dist = N_max * (N_max - 1) / 2;
          double *dist   = NULL;
          double  d_min  = 0;
          double  d_max  = 0;
          double *radius = NULL;

    dist = calloc (n_dist, sizeof (double));
    if (dist == NULL)
    {
        fprintf (stderr, "Out of memory while allocating `dist` in cdim");
        return -1;
    }

    radius = calloc (n_radius, sizeof (double));
    if (dist == NULL)
    {
        fprintf (stderr, "Out of memory while allocating `radius` in cdim");
        return -1;
    }

    comsar_fractal_embedding (x, N_max, m_dim, delay, n_dist, dist, &d_min, &d_max);

    for (size_t i = 0; i < n_radius; i++)
    {
        double ld_min = log (d_min);
        double ld_max = log (d_max);
        double lr_inc = (ld_min - ld_max) / (double) (n_radius - 1);

        radius[i] = exp (ld_min + i * lr_inc);
    }

    comsar_fractal_correlation_sum (n_radius, radius, n_dist, dist, N_max, Cr);

    free (dist);
    return 0;
}

/* Compute an estimate of the correlation dimension using Bader style
 *
 * This implementation is an improovement of the original Bader style
 * algorithm ONLY IN TERMS OF SPEED and readability.
 *
 * The algorithm has beyond that several issues that are addressed where
 * they occure.
 */
double
corr_dim_bader (const short *snd, const size_t delay, const size_t m_dim,
        const size_t n_bins, const size_t scaling_size)
{
    /* arbitrarily set boundary condition for distance matrix computation */
    const size_t bound = 10;

    /* arbitrarily set number of samples to consume form the input array 
     * If the input array has less than ``n_samples`` frames the behaviour
     * of this function is undefined. */
    const size_t n_samples = 2400;

    size_t n_dists = (n_samples-bound) * (n_samples-bound+1) / 2;
    double dist_min = 1.0;
    double dist_max = 0.0;

    double *dists = calloc (n_dists, sizeof (double));
    size_t *corr_hist = calloc (n_bins, sizeof (size_t));
    size_t *corr_sums = calloc (n_bins, sizeof (size_t));

    if (corr_hist == NULL || corr_sums == NULL || dists == NULL) {
        fprintf (stderr, "Failed to allocate memory.");
        return -1.0;
    }

    /* The below block is intended to compute the distances in a
     * `m_dim`-dimensional delay embedding by traversing the condensed
     * distance matrix of the embedded vectors.
     *
     * It does, however, compute the distances in the upper right triangle
     * of the distance matrix. The outcome is the values on the main diagonal
     * of the distance matrix are computed even though they equal 0 by
     * definiton. Moreover, the remaining distances are computed twice, i. e.,
     * the vectors at (n, m) and (m, n) are computed. Additionlly, many other
     * distances are omitted.
     */
    for (size_t i = 0, cnt = 0; i < n_samples-bound; i++)
    {
        for (size_t j = 0; j < n_samples-bound-i; j++)
        {
            for (size_t m = 0; m < m_dim; m++)
            {
                double diff = (double) (snd[i+m*delay] - snd[i+j+m*delay]);
                dists[cnt] += diff * diff;
            }
            dists[cnt] = sqrt (dists[cnt]);
            if (dists[cnt] > dist_max)
            {
                dist_max = dists[cnt];
            }
            cnt++;
        }
    }

   size_t bin_spacing = (size_t) (dist_max / 1000.0);
   size_t step_size   = bin_spacing == 0 ? 1 : bin_spacing;
    for (size_t i = 0; i < n_dists; i++)
    {
        if (dists[i] < dist_min)
        {
            corr_hist[0]++;
        }
        else
        {
            size_t idx = ((size_t) dists[i] - dist_min) / step_size;
            if (idx + 2 < n_bins)
            {
                corr_hist[idx+1]++;
            }
        }
    }

    /* Compute the correlation sum as the cummulative sum over
     * the correlation histogram `corr_hist`.
     * Note that the below implementation is wrong. Because of the
     * condition `i < j`, it skips the first index and ommits the last.
     * Hence, `corr_sums[0]` is always 0.
     * To correct this implementation use either `j <= i`, or `j < i+1`.*/
    for (size_t i = 0; i < n_bins; i++)
    {
        for (size_t j = 0; j < i+1; j++)
        {
            corr_sums[i] += corr_hist[j];
        }
        // printf ("cs[%zu] = %zu\n", i, corr_sums[i]);
    }

    /* Find the bin with the most points in it and its index */
    size_t max_pts = 0;
    size_t max_bin = 0;
    for (size_t i = 0; i < (size_t) ((double) n_bins * 3. / 5.); i++)
    {
        if (corr_hist[i] > max_pts)
        {
            max_pts = corr_hist[i];
            max_bin = i;
        }
    }

    /* Compute the slope */
    double x1 = log ((double) (max_bin * step_size) + (double) dist_min);
    double x2 = log ((double) ((max_bin + scaling_size) * step_size) + (double) dist_min);
    double y1 = log ((double) corr_sums[max_bin] / (double) n_dists);
    double y2 = log ((double) corr_sums[max_bin+scaling_size] / (double) n_dists);
    /*
    printf("x1: %f\nx2: %f\n", x1, x2);
    printf("y1: %f\ny2: %f\n", y1, y2);
    printf("corr_sums[max_bin]: %f\n", corr_sums[max_bin]);
    printf("n_dists: %f\n", (double)n_dists);
    printf("max_bin: %zu\n", max_bin);
    */
    free (dists);
    free (corr_hist);
    free (corr_sums);

    return (y2 - y1) / (x2 - x1);
}
