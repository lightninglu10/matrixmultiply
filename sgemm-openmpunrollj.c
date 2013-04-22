#include <stdio.h>
#include <omp.h>
#include <nmmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
    #pragma omp parallel
    {
	__m128 product, ATvect, ATvect1, ATvect2, ATvect3, ATvect4, ATvect5, ATvect6, ATvect7, Cvect;
	int id, i, k, l;
	float AT, AT1, AT2, AT3, AT4, AT5, AT6, AT7, ATj, ATj1, ATj2, ATj3, ATj4, ATj5, ATj6, ATj7;
	#pragma omp for
    for ( int j = 0; j < n/2 * 2; j += 2 ) {
	    for ( k = 0; k < m/8 * 8; k += 8 ) {
			AT = A[j*(n+1)+(k)*(n)];
			AT1 = A[j*(n+1)+(k+1)*(n)];
			AT2 = A[j*(n+1)+(k+2)*(n)];
			AT3 = A[j*(n+1)+(k+3)*(n)];
			AT4 = A[j*(n+1)+(k+4)*(n)];
			AT5 = A[j*(n+1)+(k+5)*(n)];
			AT6 = A[j*(n+1)+(k+6)*(n)];
			AT7 = A[j*(n+1)+(k+7)*(n)];

			ATj = A[(j + 1)*(n+1)+(k)*(n)];
			ATj1 = A[(j + 1)*(n+1)+(k+1)*(n)];
			ATj2 = A[(j + 1)*(n+1)+(k+2)*(n)];
			ATj3 = A[(j + 1)*(n+1)+(k+3)*(n)];
			ATj4 = A[(j + 1)*(n+1)+(k+4)*(n)];
			ATj5 = A[(j + 1)*(n+1)+(k+5)*(n)];
			ATj6 = A[(j + 1)*(n+1)+(k+6)*(n)];
			ATj7 = A[(j + 1)*(n+1)+(k+7)*(n)];

			for ( i = 0; i < n; i++ ) {
			    C[i+j*n] += (A[i+(k)*(n)] * AT) + (A[i+(k+1)*(n)] * AT1)
				+ (A[i+(k+2)*(n)] * AT2) + (A[i+(k+3)*(n)] * AT3)
				+ (A[i+(k+4)*(n)] * AT4) + (A[i+(k+5)*(n)] * AT5)
				+ (A[i+(k+6)*(n)] * AT6) + (A[i+(k+7)*(n)] * AT7);

				C[i+(j + 1)*n] += (A[i+(k)*(n)] * ATj) + (A[i+(k+1)*(n)] * ATj1)
				+ (A[i+(k+2)*(n)] * ATj2) + (A[i+(k+3)*(n)] * ATj3)
				+ (A[i+(k+4)*(n)] * ATj4) + (A[i+(k+5)*(n)] * ATj5)
				+ (A[i+(k+6)*(n)] * ATj6) + (A[i+(k+7)*(n)] * ATj7);
			}
		}
		for ( k = m/8 * 8; k < m; k++ ) {
				AT = A[j*(n+1)+(k)*(n)];
				ATj = A[(j+1)*(n+1)+(k)*(n)];
				for ( i = 0; i < n; i++ ) {
				    C[i+j*n] += A[i+(k)*(n)] * AT;
				    C[i+(j+1)*n] += A[i+(k)*(n)] * ATj;
				}
			}
	    }
	    for (int j = n/2; j < n; j += 1) {
	    	for ( k = 0; k < m/8 * 8; k += 8 ) {
				AT = A[j*(n+1)+(k)*(n)];
				AT1 = A[j*(n+1)+(k+1)*(n)];
				AT2 = A[j*(n+1)+(k+2)*(n)];
				AT3 = A[j*(n+1)+(k+3)*(n)];
				AT4 = A[j*(n+1)+(k+4)*(n)];
				AT5 = A[j*(n+1)+(k+5)*(n)];
				AT6 = A[j*(n+1)+(k+6)*(n)];
				AT7 = A[j*(n+1)+(k+7)*(n)];
				for ( i = 0; i < n; i++ ) {
				    C[i+j*n] += (A[i+(k)*(n)] * AT) + (A[i+(k+1)*(n)] * AT1)
					+ (A[i+(k+2)*(n)] * AT2) + (A[i+(k+3)*(n)] * AT3)
					+ (A[i+(k+4)*(n)] * AT4) + (A[i+(k+5)*(n)] * AT5)
					+ (A[i+(k+6)*(n)] * AT6) + (A[i+(k+7)*(n)] * AT7);
				}
			}
		    for ( k = m/8 * 8; k < m; k++ ) {
				AT = A[j*(n+1)+(k)*(n)];
				for ( i = 0; i < n; i++ ) {
				    C[i+j*n] += A[i+(k)*(n)] * AT;
				}
			}
		}	
	}
}