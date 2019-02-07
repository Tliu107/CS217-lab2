/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 256
#define TILE_WIDTH 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Cvalue = 0;

    for (int p = 0; p < (k-1) / TILE_WIDTH + 1; ++p) {  
        // loading of M and N tiles into shared memory
        if(Row < m && p * TILE_WIDTH+tx < k) {  // Boundary condition for M tile
            ds_A[ty][tx] = A[Row * k + p * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.0;
        }
        if (p*TILE_WIDTH+ty < k && Col < n) {  // Boundary condition for N tile
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * n + Col];
        } else {
            ds_B[ty][tx] = 0.0;
        }
        __syncthreads();
        
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
           
            __syncthreads();
        }
    }
    if (Row < m && Col < n) {
         C[Row * n + Col] = Cvalue;
    }
}



void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 DimGrid((n-1)/TILE_WIDTH + 1, (m-1)/TILE_WIDTH + 1, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);



    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<DimGrid,DimBlock>>>(m, n, k, A, B, C);  



}


