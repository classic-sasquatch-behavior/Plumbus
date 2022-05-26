#include"../../../cuda_includes.h"











__global__ void update_critereon_matrix_kernel(cv::cuda::PtrStepSzf responsibility_matrix, cv::cuda::PtrStepSzf availibility_matrix, cv::cuda::PtrStepSzf critereon_matrix, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }

	critereon_matrix(row, col) = responsibility_matrix(row, col) + availibility_matrix(row, col);
}









void update_critereon_matrix_launch(cv::cuda::GpuMat& responsibility_matrix, cv::cuda::GpuMat& availibility_matrix, cv::cuda::GpuMat& critereon_matrix, int N) {


	unsigned int grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int block_dim_xy = 32;

	dim3 num_blocks = { grid_dim_xy,grid_dim_xy,1 };
	dim3 threads_per_block = { block_dim_xy,block_dim_xy,1 };

	update_critereon_matrix_kernel << <num_blocks, threads_per_block >> > (responsibility_matrix, availibility_matrix, critereon_matrix, N);
	cudaDeviceSynchronize();
}