#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>


void sgemm( int m, int n, int d, float *A, float *C ) {
    __m128 vect, ATmatrix, ATmatrix1, ATmatrix2, ATmatrix3, ATmatrix4, ATmatrix5, ATmatrix6, ATmatrix7, Cmatrix;

    int k, i, j;

	for(k = 0; k < m; k++) {
		if (n == 40 && m == 48) {
			for(j = 0; j < n/8 * 8; j+= 8) {
				ATmatrix = _mm_load1_ps(A + ((j) * (n + 1) + (k) * (n)));
			    ATmatrix1 =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k) * (n)));
			    ATmatrix2 = _mm_load1_ps(A + ((j + 2) * (n + 1) + (k) * (n)));
			    ATmatrix3 =  _mm_load1_ps(A + ((j + 3) * (n + 1) + (k) * (n)));
			    ATmatrix4 = _mm_load1_ps(A + ((j+4) * (n + 1) + (k) * (n)));
			    ATmatrix5 =  _mm_load1_ps(A + ((j + 5) * (n + 1) + (k) * (n)));
			    ATmatrix6 = _mm_load1_ps(A + ((j + 6) * (n + 1) + (k) * (n)));
			    ATmatrix7 =  _mm_load1_ps(A + ((j + 7) * (n + 1) + (k) * (n)));
			    for(i = 0; i < n/8 * 8; i += 8) {
			    	float *p = A + i + k * n;

					Cmatrix = _mm_loadu_ps(C + i + j * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + j * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + j * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + j * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps(C + i + (j + 1) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 1) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 1) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 1) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps(C + i + (j + 2) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 2) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 2) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 2) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps(C + i + (j + 3) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 3) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 3) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 3) * n) + 4, Cmatrix);

					//4-8

					Cmatrix = _mm_loadu_ps(C + i + (j + 4) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 4) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 4) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 4) * n) + 4, Cmatrix);

					//5

					Cmatrix = _mm_loadu_ps(C + i + (j + 5) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 5) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 5) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 5) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps(C + i + (j + 6) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 6) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 6) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 6) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps(C + i + (j + 7) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(p), ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 7) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 7) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((p) + 4), ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 7) * n) + 4, Cmatrix);

			    }
			}
		} else {
			for(j = 0; j < n; j++) {
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
					
			    }
			    for (i = n/8 * 8; i < n; i ++) {
					C[i+j*n] += A[i+(k)*(n)] * A[j*(n+1)+(k)*(n)];
			    }
			}
		}
	}
}