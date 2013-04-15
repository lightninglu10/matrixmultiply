#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>


void sgemm( int m, int n, int d, float *A, float *C ) {
    __m128 vect, ATmatrix, ATmatrix1, ATmatrix2, ATmatrix3, ATmatrix4, ATmatrix5, ATmatrix6, ATmatrix7, ATmatrix8, ATmatrix9, Cmatrix, AMatrix, AMatrix1, AMatrix2, AMatrix3, AMatrix4;

    int k, i, j;

	for(k = 0; k < m; k++) {
		if (n == 40 && m == 48) {
			for(j = 0; j < n/8 * 8; j+= 8) {
				ATmatrix = _mm_load1_ps(A + ((j) * (n + 1) + (k) * (n)));
			    ATmatrix1 =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k) * (n)));
			    ATmatrix2 = _mm_load1_ps(A + ((j + 2) * (n + 1) + (k) * (n)));
			    ATmatrix3 =  _mm_load1_ps(A + ((j + 3) * (n + 1) + (k) * (n)));
			    ATmatrix4 = _mm_load1_ps(A + ((j + 4) * (n + 1) + (k) * (n)));
			    ATmatrix5 =  _mm_load1_ps(A + ((j + 5) * (n + 1) + (k) * (n)));
			    ATmatrix6 = _mm_load1_ps(A + ((j + 6) * (n + 1) + (k) * (n)));
			    ATmatrix7 =  _mm_load1_ps(A + ((j + 7) * (n + 1) + (k) * (n)));
/*			    ATmatrix8 = _mm_load1_ps(A + ((j + 8) * (n + 1) + (k) * (n)));
			    ATmatrix9 =  _mm_load1_ps(A + ((j + 9) * (n + 1) + (k) * (n)));*/
			    for(i = 0; i < n/20 * 20; i += 20) {

			    	AMatrix = _mm_loadu_ps(A + i + k * n);
			    	AMatrix1 = _mm_loadu_ps((A + i + k * n) + 4);
			    	AMatrix2 = _mm_loadu_ps((A + i + k * n) + 8);
			    	AMatrix3 = _mm_loadu_ps((A + i + k * n) + 12);
			    	AMatrix4 = _mm_loadu_ps((A + i + k * n) + 16);


					Cmatrix = _mm_loadu_ps(C + i + j * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + i + j * n, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + j * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + j * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + j * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + j * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + j * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + j * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + j * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + j * n) + 16, Cmatrix);


					//j = 1

					Cmatrix = _mm_loadu_ps(C + i + (j + 1) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 1) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 1) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 1) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 1) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 1) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 1) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 1) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 1) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix1);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 1) * n) + 16, Cmatrix);

					//j = 2

					Cmatrix = _mm_loadu_ps(C + i + (j + 2) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 2) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 2) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 2) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 2) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 2) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 2) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 2) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 2) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix2);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 2) * n) + 16, Cmatrix);

					//j = 3

					Cmatrix = _mm_loadu_ps(C + i + (j + 3) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 3) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 3) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 3) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 3) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 3) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 3) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 3) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 3) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix3);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 3) * n) + 16, Cmatrix);

					//j = 4

					Cmatrix = _mm_loadu_ps(C + i + (j + 4) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 4) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 4) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 4) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 4) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 4) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 4) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 4) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 4) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix4);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 4) * n) + 16, Cmatrix);

					//j = 5

					Cmatrix = _mm_loadu_ps(C + i + (j + 5) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 5) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 5) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 5) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 5) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 5) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 5) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 5) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 5) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix5);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 5) * n) + 16, Cmatrix);

					//j = 6

					Cmatrix = _mm_loadu_ps(C + i + (j + 6) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 6) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 6) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 6) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 6) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 6) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 6) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 6) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 6) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix6);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 6) * n) + 16, Cmatrix);

					//j = 7

					Cmatrix = _mm_loadu_ps(C + i + (j + 7) * n);
					vect = _mm_mul_ps(AMatrix, ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 7) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 7) * n) + 4);
					vect = _mm_mul_ps(AMatrix1, ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 7) * n) + 4, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 7) * n) + 8);
					vect = _mm_mul_ps(AMatrix2, ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 7) * n) + 8, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 7) * n) + 12);
					vect = _mm_mul_ps(AMatrix3, ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 7) * n) + 12, Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 7) * n) + 16);
					vect = _mm_mul_ps(AMatrix4, ATmatrix7);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 7) * n) + 16, Cmatrix);

/*					//j = 8

					Cmatrix = _mm_loadu_ps(C + i + (j + 8) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(A + i + k * n), ATmatrix8);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 8) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 8) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((A + i + k * n) + 4), ATmatrix8);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 8) * n) + 4, Cmatrix);

					//j = 9

					Cmatrix = _mm_loadu_ps(C + i + (j + 9) * n);
					vect = _mm_mul_ps(_mm_loadu_ps(A + i + k * n), ATmatrix9);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps(C + (i + (j + 9) * n), Cmatrix);

					Cmatrix = _mm_loadu_ps((C + i + (j + 9) * n) + 4);
					vect = _mm_mul_ps(_mm_loadu_ps((A + i + k * n) + 4), ATmatrix9);
					Cmatrix = _mm_add_ps(Cmatrix, vect);
					_mm_storeu_ps((C + i + (j + 9) * n) + 4, Cmatrix);*/

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