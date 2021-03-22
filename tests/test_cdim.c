#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
            printf ("%f\n", dists[flat_idx]);
        }
    }
}


int
main (void) {
    double sig[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t delay = 2;
    size_t m_dim = 3;
    size_t len_sig = sizeof sig / sizeof (double);
    size_t n_vectors = len_sig - ((m_dim - 1) * delay);
    size_t n_dists = n_vectors * (n_vectors-1) / 2;
    double *dists = calloc (n_dists, sizeof (double));

    printf ("nv: %zu\nndists: %zu\n" , n_vectors, n_dists);
    delay_embedding_dists (sig, n_vectors, delay, m_dim, dists);

    for (size_t i=0; i<n_dists; i++)
    {
        printf ("%f\n", *(dists+i));
    }
    free (dists);
    return 0;
}
    
