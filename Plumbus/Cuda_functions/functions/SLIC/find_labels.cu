#include"../../../cuda_includes.h"
#include"../../../config.h"









__global__ void find_labels_kernel(iptr src_L, iptr src_A, iptr src_B, iptr labels, iptr row_vals, iptr col_vals, iptr sector_LUT, int density, int S, int K_rows, int K_cols ) {

	//problem is in here: labels are being set too high and too low
	int thread_row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int thread_col = (blockIdx.x * blockDim.x) + threadIdx.x;

	int pixel_rows = labels.rows;
	int pixel_cols = labels.cols;

	if (thread_row >= pixel_rows || thread_col >= pixel_cols) { return; }

	//this might be wrong, think it through
	int sector_row = (thread_row - (thread_row % S))/S;
	int sector_col = (thread_col - (thread_col % S)) / S;
	int sector_id = (sector_row * K_cols) + sector_col;
	int LUT_index = sector_id * 9;

	int adjacent_centers[9];
	for (int center = 0; center < 9; center++) {
		adjacent_centers[center] = sector_LUT(0, LUT_index + center);
	}

	int center_pixel_coordinates[9][2];

	for (int target_center = 0; target_center < 9; target_center++) {
		int target_center_id = adjacent_centers[target_center]; //should end up containing the given center's center id

		if (target_center_id == -1) {
			center_pixel_coordinates[target_center][0] = -1;
			center_pixel_coordinates[target_center][1] = -1;
		}
		else {
			center_pixel_coordinates[target_center][0] = row_vals(0, target_center_id);
			center_pixel_coordinates[target_center][1] = col_vals(0, target_center_id);
		}
	}


	int thread_L = src_L(thread_row, thread_col); 
	int thread_A = src_A(thread_row, thread_col); 
	int thread_B = src_B(thread_row, thread_col); 
	int thread_color[3] = { thread_L, thread_A, thread_B };

	int closest_center_id = -1;
	int closest_center_distance = INF;

//check all valid centers, determine which is nearest
	for (int center = 0; center < 9; center++) { 

		int center_id = adjacent_centers[center];

		int center_pixel_row = center_pixel_coordinates[center][0]; 
		int center_pixel_col = center_pixel_coordinates[center][1];

		if (center_pixel_row == -1 || center_pixel_col == -1) { break; }

		int center_L = src_L(center_pixel_row, center_pixel_col);
		int center_A = src_A(center_pixel_row, center_pixel_col);
		int center_B = src_B(center_pixel_row, center_pixel_col);
		int center_color[3] = { center_L, center_A, center_B };

		//perform distance check
		int channel_diff_sum = 0;
		for (int channel = 0; channel < 3; channel++) {
			int channel_diff = thread_color[channel] - center_color[channel];
			channel_diff_sum += channel_diff * channel_diff;
		}

		int dlab = sqrtf(channel_diff_sum); 
		int dxy = sqrtf(((thread_row - center_pixel_row)*(thread_row - center_pixel_row)) + ((thread_col - center_pixel_col)*(thread_col - center_pixel_col)));
		float xy_mod = density/S;

		int distance_to_center = dlab + (xy_mod * dxy); //float gets forcibly converted to int here, probably not great

		if (distance_to_center < closest_center_distance) {
			closest_center_distance = distance_to_center;
			closest_center_id = center_id;
		}
	}

	labels(thread_row, thread_col) = closest_center_id; 
}




void find_labels_launch(gMat& src_L, gMat& src_A, gMat& src_B, gMat& labels, gMat& row_vals, gMat& col_vals, gMat& sector_LUT, int density, int S, int K_rows, int K_cols, int K) {


	unsigned int pixel_rows = labels.rows;
	unsigned int pixel_cols = labels.cols;

	unsigned int block_dim_xy = 32;
	unsigned int grid_dim_x = ((pixel_cols - (pixel_cols % block_dim_xy)) / block_dim_xy) + 1;
	unsigned int grid_dim_y = ((pixel_rows - (pixel_rows % block_dim_xy)) / block_dim_xy) + 1;


	dim3 num_blocks(grid_dim_x, grid_dim_y, 1);
	dim3 threads_per_block(block_dim_xy, block_dim_xy, 1);


	find_labels_kernel << <num_blocks, threads_per_block >>> (src_L, src_A, src_B, labels, row_vals, col_vals, sector_LUT, density, S, K_rows, K_cols);
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s: %s \n", cudaGetErrorString(error), "find labels");
	}


}