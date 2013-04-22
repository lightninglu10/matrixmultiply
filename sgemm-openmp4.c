#include <stdio.h>
#include <omp.h>
#include <nmmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
	#pragma omp parallel
	{
		__m128 vect, ATmatrix, ATvect1, ATvect2, ATvect3, ATvect4, ATvect5, ATvect6, ATvect7, Cmatrix, vect2, vect3, vect4, vect5, vect6, vect7, vect8, ATjmatrix, ATvect1j, ATvect2j, ATvect3j, ATvect4j, ATvect5j, ATvect6j, ATvect7j;
		#pragma omp for
		for( int j = 0; j < n/2; j+=2 ) {
			for( int k = 0; k < m/8 * 8; k+= 8 ) {
				ATmatrix =  _mm_load1_ps(A + (j * (n + 1) + (k) * (n)));
				float AT = A[j*(n+1)+k*(n)];

				ATvect1 =  _mm_load1_ps(A + (j * (n + 1) + (k+1) * (n)));
				float AT1 = A[j*(n+1)+(k+1)*(n)];

				ATvect2 =  _mm_load1_ps(A + (j * (n + 1) + (k+2) * (n)));
				float AT2 = A[j*(n+1)+(k+2)*(n)];

				ATvect3 =  _mm_load1_ps(A + (j * (n + 1) + (k+3) * (n)));
				float AT3 = A[j*(n+1)+(k+3)*(n)];

				ATvect4 =  _mm_load1_ps(A + (j * (n + 1) + (k+4) * (n)));
				float AT4 = A[j*(n+1)+(k+4)*(n)];

				ATvect5 =  _mm_load1_ps(A + (j * (n + 1) + (k+5) * (n)));
				float AT5 = A[j*(n+1)+(k+5)*(n)];

				ATvect6 =  _mm_load1_ps(A + (j * (n + 1) + (k+6) * (n)));
				float AT6 = A[j*(n+1)+(k+6)*(n)];

				ATvect7 =  _mm_load1_ps(A + (j * (n + 1) + (k+7) * (n)));
				float AT7 = A[j*(n+1)+(k+7)*(n)];

				//j + 1

				ATjmatrix =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k) * (n)));
				float ATj = A[(j + 1)*(n+1)+k*(n)];

				ATvect1j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+1) * (n)));
				float AT1j = A[(j + 1)*(n+1)+(k+1)*(n)];

				ATvect2j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+2) * (n)));
				float AT2j = A[(j + 1)*(n+1)+(k+2)*(n)];

				ATvect3j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+3) * (n)));
				float AT3j = A[(j + 1)*(n+1)+(k+3)*(n)];

				ATvect4j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+4) * (n)));
				float AT4j = A[(j + 1)*(n+1)+(k+4)*(n)];

				ATvect5j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+5) * (n)));
				float AT5j = A[(j + 1)*(n+1)+(k+5)*(n)];

				ATvect6j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+6) * (n)));
				float AT6j = A[(j + 1)*(n+1)+(k+6)*(n)];

				ATvect7j =  _mm_load1_ps(A + ((j + 1) * (n + 1) + (k+7) * (n)));
				float AT7j = A[(j + 1)*(n+1)+(k+7)*(n)];

				for( int i = 0; i < n/8 * 8; i+= 8 ) {
					float *temp = C + i + j * n;
					float *tempj = C + i + (j + 1) * n;
					float *tmp = A + i + (k)*(n);
					float *tmp1 = A + i + (k + 1)*(n);
					float *tmp2 = A + i + (k + 2)*(n);
					float *tmp3 = A + i + (k + 3)*(n);
					float *tmp4 = A + i + (k + 4)*(n);
					float *tmp5 = A + i + (k + 5)*(n);
					float *tmp6 = A + i + (k + 6)*(n);
					float *tmp7 = A + i + (k + 7)*(n);

					Cmatrix = _mm_loadu_ps(temp);
				    vect = _mm_mul_ps(_mm_loadu_ps(tmp), ATmatrix);
				    vect2 = _mm_mul_ps(_mm_loadu_ps(tmp1), ATvect1);
				    vect3 = _mm_mul_ps(_mm_loadu_ps(tmp2), ATvect2);
				    vect4 = _mm_mul_ps(_mm_loadu_ps(tmp3), ATvect3);
				    vect5 = _mm_mul_ps(_mm_loadu_ps(tmp4), ATvect4);
				    vect6 = _mm_mul_ps(_mm_loadu_ps(tmp5), ATvect5);
				    vect7 = _mm_mul_ps(_mm_loadu_ps(tmp6), ATvect6);
				    vect8 = _mm_mul_ps(_mm_loadu_ps(tmp7), ATvect7);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    Cmatrix = _mm_add_ps(Cmatrix, vect2);
				    Cmatrix = _mm_add_ps(Cmatrix, vect3);
				    Cmatrix = _mm_add_ps(Cmatrix, vect4);
				    Cmatrix = _mm_add_ps(Cmatrix, vect5);
				    Cmatrix = _mm_add_ps(Cmatrix, vect6);
				    Cmatrix = _mm_add_ps(Cmatrix, vect7);
				    Cmatrix = _mm_add_ps(Cmatrix, vect8);
				    _mm_storeu_ps(temp, Cmatrix);
				    //j + 1
				    Cmatrix = _mm_loadu_ps(tempj);
				    vect = _mm_mul_ps(_mm_loadu_ps(tmp), ATjmatrix);
				    vect2 = _mm_mul_ps(_mm_loadu_ps(tmp1), ATvect1j);
				    vect3 = _mm_mul_ps(_mm_loadu_ps(tmp2), ATvect2j);
				    vect4 = _mm_mul_ps(_mm_loadu_ps(tmp3), ATvect3j);
				    vect5 = _mm_mul_ps(_mm_loadu_ps(tmp4), ATvect4j);
				    vect6 = _mm_mul_ps(_mm_loadu_ps(tmp5), ATvect5j);
				    vect7 = _mm_mul_ps(_mm_loadu_ps(tmp6), ATvect6j);
				    vect8 = _mm_mul_ps(_mm_loadu_ps(tmp7), ATvect7j);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    Cmatrix = _mm_add_ps(Cmatrix, vect2);
				    Cmatrix = _mm_add_ps(Cmatrix, vect3);
				    Cmatrix = _mm_add_ps(Cmatrix, vect4);
				    Cmatrix = _mm_add_ps(Cmatrix, vect5);
				    Cmatrix = _mm_add_ps(Cmatrix, vect6);
				    Cmatrix = _mm_add_ps(Cmatrix, vect7);
				    Cmatrix = _mm_add_ps(Cmatrix, vect8);
				    _mm_storeu_ps(tempj, Cmatrix);

				    // i = 1

				    Cmatrix = _mm_loadu_ps((temp) + 4);
				    vect = _mm_mul_ps(_mm_loadu_ps((tmp) + 4), ATmatrix);
				    vect2 = _mm_mul_ps(_mm_loadu_ps((tmp1) + 4), ATvect1);
				    vect3 = _mm_mul_ps(_mm_loadu_ps((tmp2) + 4), ATvect2);
				    vect4 = _mm_mul_ps(_mm_loadu_ps((tmp3) + 4), ATvect3);
				    vect5 = _mm_mul_ps(_mm_loadu_ps(tmp4 + 4), ATvect4);
				    vect6 = _mm_mul_ps(_mm_loadu_ps(tmp5 + 4), ATvect5);
				    vect7 = _mm_mul_ps(_mm_loadu_ps(tmp6 + 4), ATvect6);
				    vect8 = _mm_mul_ps(_mm_loadu_ps(tmp7 + 4), ATvect7);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    Cmatrix = _mm_add_ps(Cmatrix, vect2);
				    Cmatrix = _mm_add_ps(Cmatrix, vect3);
				    Cmatrix = _mm_add_ps(Cmatrix, vect4);
				    Cmatrix = _mm_add_ps(Cmatrix, vect5);
				    Cmatrix = _mm_add_ps(Cmatrix, vect6);
				    Cmatrix = _mm_add_ps(Cmatrix, vect7);
				    Cmatrix = _mm_add_ps(Cmatrix, vect8);
				    _mm_storeu_ps((temp) + 4, Cmatrix);
				    // j + 1
				    Cmatrix = _mm_loadu_ps(tempj + 4);
				    vect = _mm_mul_ps(_mm_loadu_ps(tmp + 4), ATjmatrix);
				    vect2 = _mm_mul_ps(_mm_loadu_ps(tmp1 + 4), ATvect1j);
				    vect3 = _mm_mul_ps(_mm_loadu_ps(tmp2 + 4), ATvect2j);
				    vect4 = _mm_mul_ps(_mm_loadu_ps(tmp3 + 4), ATvect3j);
				    vect5 = _mm_mul_ps(_mm_loadu_ps(tmp4 + 4), ATvect4j);
				    vect6 = _mm_mul_ps(_mm_loadu_ps(tmp5 + 4), ATvect5j);
				    vect7 = _mm_mul_ps(_mm_loadu_ps(tmp6 + 4), ATvect6j);
				    vect8 = _mm_mul_ps(_mm_loadu_ps(tmp7 + 4), ATvect7j);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    Cmatrix = _mm_add_ps(Cmatrix, vect2);
				    Cmatrix = _mm_add_ps(Cmatrix, vect3);
				    Cmatrix = _mm_add_ps(Cmatrix, vect4);
				    Cmatrix = _mm_add_ps(Cmatrix, vect5);
				    Cmatrix = _mm_add_ps(Cmatrix, vect6);
				    Cmatrix = _mm_add_ps(Cmatrix, vect7);
				    Cmatrix = _mm_add_ps(Cmatrix, vect8);
				    _mm_storeu_ps(tempj + 4, Cmatrix);

/*				    Cmatrix = _mm_loadu_ps((C + i + j * n) + 8);
					vect = _mm_mul_ps(_mm_loadu_ps((tmp) + 8), ATmatrix);
				    vect2 = _mm_mul_ps(_mm_loadu_ps((tmp1) + 8), ATvect1);
				    vect3 = _mm_mul_ps(_mm_loadu_ps((tmp2) + 8), ATvect2);
				    vect4 = _mm_mul_ps(_mm_loadu_ps((tmp3) + 8), ATvect3);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    Cmatrix = _mm_add_ps(Cmatrix, vect2);
				    _mm_storeu_ps((C + i + j * n) + 8, Cmatrix);*/
				}
				for (int i = n/8 * 8; i < n; i += 1) {
					C[i+j*n] += A[i+k*(n)] * AT + A[i+(k+1)*(n)] * AT1 + A[i+(k+2)*(n)] * AT2 + A[i+(k+3)*(n)] * AT3
					+ A[i+(k + 4)*(n)] * AT4 + A[i+(k+5)*(n)] * AT5 + A[i+(k+6)*(n)] * AT6 + A[i+(k+7)*(n)] * AT7;

					C[i+(j + 1) *n] += A[i+k*(n)] * ATj + A[i+(k+1)*(n)] * AT1j + A[i+(k+2)*(n)] * AT2j + A[i+(k+3)*(n)] * AT3j
					+ A[i+(k + 4)*(n)] * AT4j + A[i+(k+5)*(n)] * AT5j + A[i+(k+6)*(n)] * AT6j + A[i+(k+7)*(n)] * AT7j;
				}
			}
			for (int k = m/8 * 8; k < m; k += 1) {
				ATmatrix =  _mm_load1_ps(A + (j * (n + 1) + (k) * (n)));
				float AT = A[j*(n+1)+k*(n)];

				ATjmatrix =  _mm_load1_ps(A + ((j+1) * (n + 1) + (k) * (n)));
				float ATj = A[(j+1)*(n+1)+k*(n)];
				for( int i = 0; i < n/12 * 12; i+= 12 ) {
					float *temp = C + i + j * n;
					float *t2 = A + i + (k)*(n);
					float *tempj = C + i + (j+1) * n;
					
					Cmatrix = _mm_loadu_ps(temp);
				    vect = _mm_mul_ps(_mm_loadu_ps(t2), ATmatrix);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    _mm_storeu_ps(temp, Cmatrix);
				    //j+1
				    Cmatrix = _mm_loadu_ps(tempj);
				    vect = _mm_mul_ps(_mm_loadu_ps(t2), ATmatrix);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    _mm_storeu_ps(tempj, Cmatrix);

				    //i = 1
				    Cmatrix = _mm_loadu_ps((temp) + 4);
				    vect = _mm_mul_ps(_mm_loadu_ps((t2) + 4), ATmatrix);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    _mm_storeu_ps((temp) + 4, Cmatrix);
				    //j+1
				    Cmatrix = _mm_loadu_ps((tempj) + 4);
				    vect = _mm_mul_ps(_mm_loadu_ps((t2) + 4), ATmatrix);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    _mm_storeu_ps((tempj) + 4, Cmatrix);

				    //i = 2
				    Cmatrix = _mm_loadu_ps((temp) + 8);
				    vect = _mm_mul_ps(_mm_loadu_ps((t2) + 8), ATmatrix);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    _mm_storeu_ps((temp) + 8, Cmatrix);
				    //j + 1
				    Cmatrix = _mm_loadu_ps((tempj) + 8);
				    vect = _mm_mul_ps(_mm_loadu_ps((t2) + 8), ATmatrix);
				    Cmatrix = _mm_add_ps(Cmatrix, vect);
				    _mm_storeu_ps((tempj) + 8, Cmatrix);
				}
				for (int i = n/12 * 12; i < n; i += 1) {
					C[i+j*n] += A[i+k*(n)] * AT;
					C[i+(j+1)*n] += A[i+k*(n)] * ATj;
				}
			}
		}
	}
}
