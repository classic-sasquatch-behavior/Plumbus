#include"../../../cuda_includes.h"



__global__ void condense_labels_kernel(iptr labels, iptr row_sums, iptr col_sums, iptr num_sums) {

	int thread_row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int thread_col = (blockIdx.x * blockDim.x) + threadIdx.x;

	int pixel_rows = labels.rows;
	int pixel_cols = labels.cols;
	int num_pixels = pixel_rows * pixel_cols;

	if (thread_row >= pixel_rows || thread_col >= pixel_cols) { return; }

	int label = labels(thread_row, thread_col);

	atomicAdd(&row_sums(0, label), thread_row);
	atomicAdd(&col_sums(0, label), thread_col);
	atomicAdd(&num_sums(0, label), 1);
}








__global__ void update_centers_kernel(iptr labels, iptr row_vals, iptr col_vals, iptr row_sums, iptr col_sums, iptr num_sums, int K, int* row_displacement, int* col_displacement) {
	int center_label = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (center_label >= K) { return; }

	int row_sum = row_sums(0, center_label);
	int col_sum = col_sums(0, center_label);
	int num_constituents = num_sums(0, center_label);

	int row_result = row_sum / num_constituents;
	int col_result = col_sum / num_constituents; 

	int old_pixel_row = row_vals(0, center_label);
	int old_pixel_col = col_vals(0, center_label);

	int row_disp = abs(row_result - old_pixel_row);
	int col_disp = abs(col_result - old_pixel_col);

	row_vals(0, center_label) = row_result;
	col_vals(0, center_label) = col_result;

	atomicAdd(row_displacement, row_disp);
	atomicAdd(col_displacement, col_disp);

}











void update_centers_launch(gMat& labels, gMat& row_vals, gMat& col_vals, int* average_displacement, int K_rows, int K_cols, int K) {

	int pixel_rows = labels.rows;
	int pixel_cols = labels.cols;
	int num_pixels = pixel_rows * pixel_cols;

	cv::cuda::GpuMat row_sums(cv::Size(K, 1), CV_32SC1);
	cv::cuda::GpuMat col_sums(cv::Size(K, 1), CV_32SC1);
	cv::cuda::GpuMat num_sums(cv::Size(K, 1), CV_32SC1);



	//preparing for kernel a
	unsigned int ka_block_dim_xy = 32;
	unsigned int ka_grid_dim_x = ((pixel_cols - (pixel_cols % ka_block_dim_xy)) / ka_block_dim_xy) + 1;
	unsigned int ka_grid_dim_y = ((pixel_rows - (pixel_rows % ka_block_dim_xy)) / ka_block_dim_xy) + 1;

	dim3 ka_num_blocks(ka_grid_dim_x, ka_grid_dim_y, 1);
	dim3 ka_threads_per_block(ka_block_dim_xy, ka_block_dim_xy, 1);

	std::cout << "condensing labels..." << std::endl;
	condense_labels_kernel << <ka_num_blocks, ka_threads_per_block >> > (labels, row_sums, col_sums, num_sums);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError(); 
	if (error != cudaSuccess){printf("CUDA error: %s: %s \n", cudaGetErrorString(error), "condense labels");}



	//preparing for kernel b
	
	//make both into device pointers and pass them to kernel
	int total_row_displacement = 0;
	int total_col_displacement = 0;




	unsigned int kb_block_dim_x = 1024;
	unsigned int kb_grid_dim_x = ((K - (K % kb_block_dim_x)) / kb_block_dim_x) + 1;

	dim3 kb_num_blocks(kb_grid_dim_x, 1, 1);
	dim3 kb_threads_per_block(kb_block_dim_x, 1, 1);



	int row_displacement_sum = 0;
	int col_displacement_sum = 0;
	int* h_row_displacement_sum = &row_displacement_sum;
	int* h_col_displacement_sum = &col_displacement_sum;

	int* d_row_displacement_sum;
	int* d_col_displacement_sum;

	cudaMalloc(&d_row_displacement_sum, sizeof(int));
	cudaMalloc(&d_col_displacement_sum, sizeof(int));
	cudaMemcpy(d_row_displacement_sum, h_row_displacement_sum, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_displacement_sum, h_col_displacement_sum, sizeof(int), cudaMemcpyHostToDevice);

	std::cout << "updating centers (kernel)..." << std::endl;
	update_centers_kernel << <kb_num_blocks, kb_threads_per_block >> > (labels, row_vals, col_vals, row_sums, col_sums, num_sums, K, d_row_displacement_sum, d_col_displacement_sum);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) { printf("CUDA error: %s: %s \n", cudaGetErrorString(error), "update centers"); }

	cudaMemcpy(h_row_displacement_sum, d_row_displacement_sum, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_col_displacement_sum, d_col_displacement_sum, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_row_displacement_sum);
	cudaFree(d_col_displacement_sum);

	row_displacement_sum /= pixel_rows;
	col_displacement_sum /= pixel_cols;
	
	int total_displacement = sqrt((row_displacement_sum * row_displacement_sum) + (col_displacement_sum * col_displacement_sum));
	*average_displacement = total_displacement;
}