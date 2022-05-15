#include"../../cuda_includes.h"
#include"../../config.h"






__global__ void form_responsibility_matrix_kernel(cv::cuda::PtrStepSzi src, cv::cuda::PtrStepSzi dst, int N) {
	
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row >= N || col >= N) { return; }

	int max_similarity = -INF;
	int source_val = src(row, col);

	for (int traverse_row = 0; traverse_row < N; traverse_row++) {
		int value_at_cell = src(row, traverse_row);
		if (traverse_row != col) {
			if (value_at_cell > max_similarity) {
				max_similarity = value_at_cell;
			}
		}
	}

	int result = source_val - max_similarity;
	dst(row, col) = result;

	//dst(row,col) = src(row,col) - max of remaining similarities in row
}












void form_responsibility_matrix_launch(cv::cuda::GpuMat& source, cv::cuda::GpuMat& output, int N) {

	unsigned int grid_dim_xy = ((N - (N % 32)) / 32) + 1;
	unsigned int block_dim_xy = 32;

	dim3 num_blocks = {grid_dim_xy,grid_dim_xy,1};
	dim3 threads_per_block = {block_dim_xy,block_dim_xy,1};


	form_responsibility_matrix_kernel << <num_blocks, threads_per_block >> > (source, output, N);
	cudaDeviceSynchronize();


}