#include"../../cuda_includes.h"











__global__ void form_critereon_matrix_kernel(cv::cuda::PtrStepSzi src_a, cv::cuda::PtrStepSzi src_b, cv::cuda::PtrStepSzi dst, int N) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }

	dst(row, col) = src_a(row, col) + src_b(row, col);
}









void form_critereon_matrix_launch(cv::cuda::GpuMat& source_a, cv::cuda::GpuMat& source_b, cv::cuda::GpuMat& output, int N) {


	unsigned int grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int block_dim_xy = 32;

	dim3 num_blocks = { grid_dim_xy,grid_dim_xy,1 };
	dim3 threads_per_block = { block_dim_xy,block_dim_xy,1 };

	form_critereon_matrix_kernel << <num_blocks, threads_per_block >> > (source_a, source_b, output, N);
	cudaDeviceSynchronize();
}