#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>


void sgemm( int m, int n, int d, float *A, float *C ) {
    __m128 vect;
    __m128 ATmatrix;
    __m128 Cmatrix;

    int k, i, j;

	for(j = 0; j < n; j++) {
		for(k = 0; k < m; k++) {
		    ATmatrix =  _mm_load1_ps(A + (j * (n + 1) + (k) * (n)));
		    for(i = 0; i < n/8 * 8; i += 8) {

				Cmatrix = _mm_loadu_ps(C + i + j * n);
				vect = _mm_mul_ps(_mm_loadu_ps(A + i + (k)*(n)), ATmatrix);
				Cmatrix = _mm_add_ps(Cmatrix, vect);
				_mm_storeu_ps(C + (i + j * n), Cmatrix);

				Cmatrix = _mm_loadu_ps((C + i + j * n) + 4);
				vect = _mm_mul_ps(_mm_loadu_ps((A + i + (k)*(n)) + 4), ATmatrix);
				Cmatrix = _mm_add_ps(Cmatrix, vect);
				_mm_storeu_ps((C + i + j * n) + 4, Cmatrix);
				
				/*Cmatrix = _mm_loadu_ps((C + i + j * n) + 8);
				vect = _mm_mul_ps(_mm_loadu_ps((A + i + (k)*(n)) + 8), ATmatrix);
				Cmatrix = _mm_add_ps(Cmatrix, vect);
				_mm_storeu_ps((C + i + j * n) + 8, Cmatrix);
				
				Cmatrix = _mm_loadu_ps((C + i + j * n) + 12);
				vect = _mm_mul_ps(_mm_loadu_ps((A + i + (k)*(n)) + 12), ATmatrix);
				Cmatrix = _mm_add_ps(Cmatrix, vect);
				_mm_storeu_ps((C + i + j * n) + 12, Cmatrix);*/
		    }
		    for (i = n/8 * 8; i < n; i ++) {
				C[i+j*n] += A[i+(k)*(n)] * A[j*(n+1)+(k)*(n)];
		    }
		}
	}
}