#include"../../cuda_includes.h"
#include"cuda_function_includes.h"
#include"../../classes.h"
#include"../../config.h"



//write error checking function and so on here


#pragma structors


CudaUtil::CudaUtil() {

}

CudaUtil::~CudaUtil() {

}



#pragma endregion


//we actually do not need the mat dimensionality. stop being lazy and fix it.
void CudaUtil::get_kernel_structure(gMat& input, dim3* num_blocks_out, dim3* threads_per_block_out, int mat_dimensionality, int kernel_dimensionality) {

	int input_rows = input.rows;
	int input_cols = input.cols;
	int N = input_rows * input_cols;

	dim3 num_blocks;
	dim3 threads_per_block;

	if (mat_dimensionality == 2 && kernel_dimensionality == 2) {
		unsigned int block_dim_xy = 32;
		unsigned int grid_dim_x = ((input_cols - (input_cols % block_dim_xy)) / block_dim_xy) + 1;
		unsigned int grid_dim_y = ((input_rows - (input_rows % block_dim_xy)) / block_dim_xy) + 1;

		dim3 num_blocks(grid_dim_x, grid_dim_y, 1);
		dim3 threads_per_block(block_dim_xy, block_dim_xy, 1);
	}

	if (mat_dimensionality == 2 && kernel_dimensionality == 1) {

		unsigned int block_dim_x = 1024;
		unsigned int grid_dim_x = ((N - (N % block_dim_x)) / block_dim_x) + 1;

		dim3 num_blocks(grid_dim_x, 1, 1);
		dim3 threads_per_block(block_dim_x, 1, 1);
	}

	if (mat_dimensionality == 1 && kernel_dimensionality == 1) {
		unsigned int block_dim_x = 1024;
		unsigned int grid_dim_x = ((N - (N % block_dim_x)) / block_dim_x) + 1;

		dim3 num_blocks(grid_dim_x, 1, 1);
		dim3 threads_per_block(block_dim_x, 1, 1);
	}

	else { std::cout << "dimensions not configured. mat dimensionality: " << mat_dimensionality << ", kernel dimensionality: " << kernel_dimensionality << std::endl; }

	num_blocks_out = &num_blocks;
	threads_per_block_out = &threads_per_block;
}








