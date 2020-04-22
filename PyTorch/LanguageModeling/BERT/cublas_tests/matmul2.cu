// nvcc 001 isamax .c -lcublas
#include <iostream>
#include </usr/include/stdio.h>
#include </usr/include/stdlib.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "cublas_v2.h"
#include "curand.h"
#include "cuda_fp16.h"
#include <math.h>
#include <time.h>
#include <library_types.h>
#include <cuda.h>
#include "device_launch_parameters.h"

using namespace std;


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = __float2half(in[idx]);
   }
}
__global__ void convertABFp32ToFp16 (half *out, float *in) {
    *out = __float2half(*in);
}

__global__ void setAlpha (float *in) {
    *in = 1;
}
__global__ void setBeta (float *in) {
    *in = 0;
}

 //Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
 void print_matrix(float *A, int nr_rows_A, int nr_cols_A) {
 
     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
             std::cout << A[j * nr_rows_A + i] << " ";
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }
 void set_matrix(float *A, int nr_rows_A, int nr_cols_A) {
 
     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
             A[j * nr_rows_A + i] = 1;
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

	
void gpu_blas_mmul(half *A, half *B, half *C, int m, int k, int n, int i) {
    int lda=k,ldb=n,ldc=n;
    const float alf = 1.0f;
    const float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
		cublasStatus_t cublasStat  = cublasCreate(&handle);
		// Set the math mode to allow cuBLAS to use Tensor Cores:
		cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
/*
		if (i == 0){//compute type cuda_16
			float * alf;
			float * bet;
			cudaMalloc(&alf, sizeof(float));
			setAlpha<<<1,1>>>(alf);
			cudaMalloc(&bet, sizeof(float));
			setBeta<<<1,1>>>(bet);
			half *dalf, *dbet;
			cudaMalloc(&dalf, sizeof(half));
			cudaMalloc(&dbet, sizeof(half));
			convertABFp32ToFp16 <<< 1,1 >>> (dalf, alf);
			convertABFp32ToFp16 <<< 1,1 >>> (dbet, bet);
		  for(int i = 0; i < 10; i++){
			// Do the actual multiplication
			cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, dalf, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, dbet, C,CUDA_R_16F, ldc,CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		printf("%d", cublasStat);
		}
		cudaFree(alf);
		cudaFree(dalf);
		cudaFree(bet);
		cudaFree(dbet);
	}
		else if (i == 1){//compute type cuda_32
			float alf = 1;
			float bet = 0;
			float *alpha;
			float *beta;
			alpha = &alf;
			beta = &bet;	
		  for(int i = 0; i < 10; i++){
			// Do the actual multiplication
			cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, beta, C,CUDA_R_16F, ldc,CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		}
		}
*/

for(int i = 0; i < 10; i++){
			// Do the actual multiplication
			cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, beta, C,CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
			//cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1536, 4096, alpha, B, CUDA_R_16F, 1024, A, CUDA_R_16F, 4096, beta, C,CUDA_R_16F,1024 ,CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

/*
		for(int i = 0; i < 20; i++){
			cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, CUDA_R_16F, lda, 384 * 384, B, CUDA_R_16F, ldb, 384 * 64, beta, C, CUDA_R_16F, ldc, 384 * 64, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		}
*/
    // Destroy the handle
    cublasDestroy(handle);
}

int main() {
    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    // for simplicity we are going to use square arrays
    nr_rows_A = 4096;
		nr_cols_A = 4096;
		nr_rows_B = 4096;
		nr_cols_B = 4096;
		nr_rows_C = 4096;
		nr_cols_C = 4096;

		//array on device of type half.
		float *df_A, *df_B, *df_C;	
    // Allocate 3 arrays on GPU
    half *d_A, *d_B, *d_C;

    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(half));
    cudaMalloc(&df_A,nr_rows_A * nr_cols_A * sizeof(float));
		GPU_fill_rand(df_A, nr_rows_A, nr_cols_A);	
		convertFp32ToFp16 <<< 1,1 >>> (d_A, df_A, nr_rows_A * nr_cols_A);
    
		cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(half));
    cudaMalloc(&df_B,nr_rows_B * nr_cols_B * sizeof(float));
		GPU_fill_rand(df_B, nr_rows_B, nr_cols_B);	
		convertFp32ToFp16 <<< 1,1 >>> (d_B, df_B, nr_rows_B * nr_cols_B);
    
		cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(half));
    cudaMalloc(&df_C,nr_rows_C * nr_cols_C * sizeof(float));
   
	  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B, 0);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(df_A);
    cudaFree(df_B);
    cudaFree(df_C);


    return 0;
}

