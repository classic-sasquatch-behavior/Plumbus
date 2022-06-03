#include"../../../cuda_includes.h"
#include"../../../config.h"







__global__ void pop_zero_vec_kernel(iptr input, int N) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (x >= N) {return;}














}












void pop_zero_vec_launch(gMat& input) {

	//vector is assumed to be 1d and horizontal


	unsigned int N = input.cols;
	unsigned int block_dim_x = 1024;
	unsigned int grid_dim_x = ((N - (N % block_dim_x)) / block_dim_x) + 1;


	dim3 num_blocks(grid_dim_x, 1, 1);
	dim3 threads_per_block(block_dim_x, 1, 1);


	pop_zero_vec_kernel <<<num_blocks, threads_per_block>>> (input, N);


















}