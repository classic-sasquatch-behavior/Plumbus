#include"../../../cuda_includes.h"
#include"../../../config.h"









__global__ void find_labels_kernel(iptr src_L, iptr src_A, iptr src_B, iptr labels, iptr row_vals, iptr col_vals, iptr sector_LUT, int density, int k_step ) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row < 0|| col < 0 || row >= src_L.rows || col >= src_L.cols) { return; }

	int sector_rows = row_vals.rows;
	int sector_cols = row_vals.cols;


	int sector_row = (row - (row % k_step))/k_step;
	int sector_col = (col - (col % k_step)) / k_step;;
	int sector_id = (sector_row * sector_cols) + sector_col;
	int closest_centers[9][2];

	for (int center = 0; center < 9; center++) {

		int center_neighbor_sector_row_id = (sector_id * 9 * 2);
		int center_neighbor_sector_col_id = (sector_id * 9 * 2) + 1;

		closest_centers[center][0] = sector_LUT(0, center_neighbor_sector_row_id);
		closest_centers[center][1] = sector_LUT(0, center_neighbor_sector_col_id);
	}

	int closest_centers_actual[9][2];
	for (int center = 0; center < 9; center++) {
		int center_sector_row = closest_centers[center][0];
		int center_sector_col = closest_centers[center][1];

		int center_actual_row = -1;
		int center_actual_col = -1;

		if (center_sector_row != -1 || center_sector_col != -1) {
			int center_actual_row = row_vals(center_sector_row, center_sector_col);
			int center_actual_col = col_vals(center_sector_row, center_sector_col);
		}

		closest_centers_actual[center][0] = center_actual_row;
		closest_centers_actual[center][1] = center_actual_col;
	}


	int focus_L = src_L(row, col); 
	int focus_A = src_A(row, col); 
	int focus_B = src_B(row, col); 
	int focus_color[3] = { focus_L, focus_A, focus_B };

	int closest_center_id = -1;
	int closest_center_distance = INF;


//check all valid centers, determine which is nearest
	for (int center = 0; center < 9; center++) {

		int center_sector_row = closest_centers[center][0];
		int center_sector_col = closest_centers[center][1];

		if (center_sector_row == -1 || center_sector_col == -1) { break; }

		int center_id = (center_sector_row * sector_cols) + center_sector_col;

		int center_actual_row = row_vals(center_sector_row, center_sector_col);
		int center_actual_col = col_vals(center_sector_row, center_sector_col);

		int center_L = src_L(center_actual_row, center_actual_col);
		int center_A = src_A(center_actual_row, center_actual_col);
		int center_B = src_B(center_actual_row, center_actual_col); 
		int center_color[3] = { center_L, center_A, center_B };

		//perform distance check
		int channel_diff_sum = 0;
		for (int channel = 0; channel < 3; channel++) {
			int channel_diff = focus_color[channel] - center_color[channel];
			channel_diff_sum += channel_diff * channel_diff;
		}

		int dlab = sqrtf(channel_diff_sum); //sqrt not allowed
		int dxy = sqrtf(((row - center_actual_row)*(row - center_actual_row)) + ((col - center_actual_col)*(col - center_actual_col))); //sqrt not allowed
		int xy_mod = density/k_step;
		int distance_to_center = dlab + (xy_mod * dxy);

		if (distance_to_center < closest_center_distance) {
			closest_center_distance = distance_to_center;
			closest_center_id = center_id;
		}
	}

	labels(row, col) = closest_center_id;
}




void find_labels_launch(cv::cuda::GpuMat& src_L, cv::cuda::GpuMat& src_A, cv::cuda::GpuMat& src_B, cv::cuda::GpuMat& labels, cv::cuda::GpuMat& row_vals, cv::cuda::GpuMat& col_vals, cv::cuda::GpuMat& sector_LUT, int density, int k_step) {

	int rows = row_vals.rows;
	int cols = row_vals.cols;

	int K = rows * cols;

	unsigned int block_dim_xy = 32;
	unsigned int grid_dim_x = ((cols - (cols % 32)) / 32) + 1;
	unsigned int grid_dim_y = ((rows - (rows % 32)) / 32) + 1;


	dim3 num_blocks(grid_dim_x, grid_dim_y, 1);
	dim3 threads_per_block(block_dim_xy, block_dim_xy, 1);


	find_labels_kernel << <num_blocks, threads_per_block >>> (src_L, src_A, src_B, labels, row_vals, col_vals, sector_LUT, density, k_step);
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}


}