#pragma once

//assumes that the exemplar mat is named src, returns row and col id. incidentally defines src_rows and src_cols
#define get_dims_ids_and_check_bounds																													\
	int src_rows = src.rows; int src_cols = src.cols;																									\
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; int col = (blockIdx.x * blockDim.x) + threadIdx.x;  int x = col; int y = row;					\
	int id = (row * src_cols) + col;																													\
	if (row >= src_rows || col >= src_cols) { return; }



//gets dims and ids without checking the bounds
#define get_dims_ids_without_bounds																														\
	int src_rows = src.rows; int src_cols = src.cols;																									\
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; int col = (blockIdx.x * blockDim.x) + threadIdx.x; int x = col; int y = row; int id = (row * src_cols) + col;								



//returns neighbor_row and neighbor_col as indices relative to row and col of thread
#define for_each_immediate_neighbor(content_of_expression)																								\
	for (int irow = -1; irow <= 1; irow++) { for (int icol = -1; icol <= 1; icol++)																		\
	{ int neighbor_row = row + irow; int neighbor_col = col + icol;																						\
	if (neighbor_row >= src_rows || neighbor_col >= src_cols || neighbor_row < 0 || neighbor_col < 0) { break; }										\
	content_of_expression																																\
	}}





//returns neighbor_row and neighbor_col as indices relative to row and col of thread
#define for_each_immediate_neighbor_specific(row_bounds, col_bounds, starting_row, starting_col, content_of_expression)																								\
	for (int irow = -1; irow <= 1; irow++) { for (int icol = -1; icol <= 1; icol++)																		\
	{ int neighbor_row = starting_row + irow; int neighbor_col = starting_col + icol;																						\
	if (neighbor_row >= row_bounds || neighbor_col >= col_bounds || neighbor_row < 0 || neighbor_col < 0) { break; }										\
	content_of_expression																																\
	}}











//simple as. leave out the semicolon and add it in the code for formatting reasons. consider making this check errors too.
#define cusync 	cudaDeviceSynchronize() 

#define cusyncerr(function_name) cudaDeviceSynchronize();	cudaError_t function_name = cudaGetLastError();												\
if (function_name != cudaSuccess) { printf("CUDA error: %s: %s \n", cudaGetErrorString(function_name), #function_name ); }



#define alias_input(actual_name_of_input) gMat &input = actual_name_of_input;
#define alias_src(actual_name_of_src) iptr &src = actual_name_of_src;

#define get_structure_from_mat																																										\
int rows, cols, N;																																													\
dim3 num_blocks, threads_per_block;																																									\
unsigned int block_dim_x, block_dim_xy, grid_dim_x, grid_dim_y, grid_dim_xy;																														\
rows = input.rows; cols = input.cols; N = rows * cols;

#define make_1d_kernel_from_structure																																								\
block_dim_x = 1024; grid_dim_x = ((N - (N%block_dim_x))/block_dim_x) + 1;																									\
num_blocks = {grid_dim_x, 1, 1}; threads_per_block = {block_dim_x, 1, 1};										

#define make_2d_kernel_from_structure																																								\
block_dim_xy = 32; grid_dim_x = ((cols - (cols%block_dim_xy))/block_dim_xy) + 1; grid_dim_y = ((rows - (rows%block_dim_xy))/block_dim_xy) + 1;				\
num_blocks = {grid_dim_x, grid_dim_y, 1}; threads_per_block = {block_dim_xy, block_dim_xy, 1};										



#define inititalize_global_flag(host_variable_name) \
int host_variable_name = 0;							 \
int* h_flag = &host_variable_name;					 \
int* d_flag;										 \
cudaMalloc(&d_flag, sizeof(int));


#define upload_global_flag cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

#define download_global_flag cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

#define destroy_global_flag cudaFree(d_flag);






