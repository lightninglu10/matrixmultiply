#include <stdio.h>
#include <omp.h>
#include <nmmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
    #pragma omp parallel
    {
	__m128 product, ATvect, ATvect1, ATvect2, ATvect3, ATvect4, ATvect5, ATvect6, ATvect7, Cvect;
	int id, i, k, l;
	float AT, AT1, AT2, AT3, AT4, AT5, AT6, AT7;
	#pragma omp for
    for ( int j = 0; j < n; j++ ) {
	    for ( k = 0; k < m/8 * 8; k += 8 ) {
			//id = omp_get_thread_num();
			AT = A[j*(n+1)+(k)*(n)];
			AT1 = A[j*(n+1)+(k+1)*(n)];
			AT2 = A[j*(n+1)+(k+2)*(n)];
			AT3 = A[j*(n+1)+(k+3)*(n)];
			AT4 = A[j*(n+1)+(k+4)*(n)];
			AT5 = A[j*(n+1)+(k+5)*(n)];
			AT6 = A[j*(n+1)+(k+6)*(n)];
			AT7 = A[j*(n+1)+(k+7)*(n)];

			for ( i = 0; i < n; i++ ) {
			    // printf("(ID=%d) i=%d, j=%d, k=%d\n", id, i, j, k);
			    C[i+j*n] += (A[i+(k)*(n)] * AT) + (A[i+(k+1)*(n)] * AT1)
				+ (A[i+(k+2)*(n)] * AT2) + (A[i+(k+3)*(n)] * AT3)
				+ (A[i+(k+4)*(n)] * AT4) + (A[i+(k+5)*(n)] * AT5)
				+ (A[i+(k+6)*(n)] * AT6) + (A[i+(k+7)*(n)] * AT7);
			}
			
		    }
		    for ( k = m/8 * 8; k < m; k++ ) {
				//id = omp_get_thread_num();
				AT = A[j*(n+1)+(k)*(n)];
				/*ATvect =  _mm_load1_ps(A + (j * (n + 1) + (k) * (n)));
				for ( i = 0; i < n/4 * 4; i += 4 ) {
				    // printf("(ID=%d) i=%d, j=%d, k=%d\n", id, i, j, k);
				    Cvect = _mm_loadu_ps(C + (i + j * n));
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k)*(n)), ATvect);
				    Cvect = _mm_add_ps(Cvect, product);
				    _mm_storeu_ps(C + (i + j * n), Cvect);
				}*/
				// for ( l = n/4 * 4; l < n; l++ ) {
				//     C[l+j*n] += A[l+(k)*(n)] * AT;
				// }
				for ( i = 0; i < n; i++ ) {
				    // printf("(ID=%d) i=%d, j=%d, k=%d\n", id, i, j, k);
				    C[i+j*n] += A[i+(k)*(n)] * AT;
				}
			}
	    }
	}
}
/*				ATvect =  _mm_load1_ps(A + (j * (n + 1) + (k) * (n)));
				ATvect1 =  _mm_load1_ps(A + (j * (n + 1) + (k + 1) * (n)));
				ATvect2 =  _mm_load1_ps(A + (j * (n + 1) + (k + 2) * (n)));
				ATvect3 =  _mm_load1_ps(A + (j * (n + 1) + (k + 3) * (n)));
				ATvect4 =  _mm_load1_ps(A + (j * (n + 1) + (k + 4) * (n)));
				ATvect5 =  _mm_load1_ps(A + (j * (n + 1) + (k + 5) * (n)));
				ATvect6 =  _mm_load1_ps(A + (j * (n + 1) + (k + 6) * (n)));
				ATvect7 =  _mm_load1_ps(A + (j * (n + 1) + (k + 7) * (n)));
				for ( i = 0; i < n/8 * 8; i += 8 ) {
				    // printf("(ID=%d) i=%d, j=%d, k=%d\n", id, i, j, k);
				    Cvect = _mm_loadu_ps(C + (i + j * n));
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k)*(n)), ATvect);
				    Cvect = _mm_add_ps(Cvect, product);
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+1)*(n)), ATvect1);
				    Cvect = _mm_add_ps(Cvect, product);
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+2)*(n)), ATvect2);
				    Cvect = _mm_add_ps(Cvect, product);	    
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+3)*(n)), ATvect3);
				    Cvect = _mm_add_ps(Cvect, product);
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+4)*(n)), ATvect4);
				    Cvect = _mm_add_ps(Cvect, product);
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+5)*(n)), ATvect5);
				    Cvect = _mm_add_ps(Cvect, product);
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+6)*(n)), ATvect6);
				    Cvect = _mm_add_ps(Cvect, product);
				    product = _mm_mul_ps(_mm_loadu_ps(A + i + (k+7)*(n)), ATvect7);
				    Cvect = _mm_add_ps(Cvect, product);
				    _mm_storeu_ps(C + (i + j * n), Cvect);

				//     Cvect = _mm_loadu_ps(C + ((i + 4) + j * n));
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k)*(n)), ATvect);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+1)*(n)), ATvect1);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+2)*(n)), ATvect2);
				//     Cvect = _mm_add_ps(Cvect, product);	    
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+3)*(n)), ATvect3);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+4)*(n)), ATvect4);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+5)*(n)), ATvect5);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+6)*(n)), ATvect6);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     product = _mm_mul_ps(_mm_loadu_ps(A + (i + 4) + (k+7)*(n)), ATvect7);
				//     Cvect = _mm_add_ps(Cvect, product);
				//     _mm_storeu_ps(C + ((i + 4) + j * n), Cvect);
				// }
				for ( l = n/8 * 8; l < n; l++ ) {
				    C[l+j*n] += (A[l+(k)*(n)] * AT) + (A[l+(k+1)*(n)] * AT1)
				        + (A[l+(k+2)*(n)] * AT2) + (A[l+(k+3)*(n)] * AT3)
					+ (A[l+(k+4)*(n)] * AT4) + (A[l+(k+5)*(n)] * AT5)
					+ (A[l+(k+6)*(n)] * AT6) + (A[l+(k+7)*(n)] * AT7);
				}*/