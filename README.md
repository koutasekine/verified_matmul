# verified_matmul

void udmatmul(int m, int n, int k, double *A, double *B, double *CU, double *CD)

F: Set of Floating Point Number
A in F^{m*n}
B in F^{n*m}

Compute matrix multiplication A*B with round up and round down
CU: Matrix multiplication A*B using round up of IEEE 754, that is, A*B <= CU with all elements
CD: Matrix multiplication A*B using round down of IEEE 754, that is, A*B >= CD with all elements

Note:
Use AVX512 in function `udmatmul`.
