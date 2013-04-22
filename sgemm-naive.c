void sgemm( int m, int n, int d, float *A, float *C )
{
  for( int k = 0; k < m; k++ )
    for( int i = 0; i < n; i++ )
      for( int j = 0; j < n; j++ )
	C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
}

void sgemm_unrolled( int m, int n, int d, float *A, float *C )
{
	int f, kn;
	int i,j,k;
  	for(k = 0; k < m; k++ ){
    	for(i = 0; i < n; i++ ) {
      		f = A[i+k*(n)];
      		kn = k*n;
      		for(j = 0; j < n/4*4; j+=4 ) {
				C[i+j    *n] += f * A[j*(n+1)+kn];
				C[i+(j+1)*n] += f * A[(j+1)*(n+1)+kn];
				C[i+(j+2)*n] += f * A[(j+2)*(n+1)+kn];
				C[i+(j+3)*n] += f * A[(j+3)*(n+1)+kn];
      		}
      		for (j = n/4*4;j < n; j++) {
      			C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
      		}
  		}
	}

}
