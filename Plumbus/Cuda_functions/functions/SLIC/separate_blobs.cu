#include"../../../cuda_includes.h"
#include"../../../config.h"



__global__ void linear_flow_kernel(iptr src, iptr temp, iptr id_LUT, int N, int* change) {
	get_dims_ids_and_check_bounds
	int id = (row * src_cols) + col;
	int self_label = src(row, col);
	int self_temp_val = temp(row, col);
	int greatest_temp_val = self_temp_val;

	#pragma unroll
	for_each_immediate_neighbor (
		int neighbor_label = src(neighbor_row, neighbor_col);
		if (self_label == neighbor_label) {
			int neighbor_temp_val = temp(neighbor_row, neighbor_col);
			if (neighbor_temp_val > greatest_temp_val) {
				greatest_temp_val = neighbor_temp_val;
			}
		}
	) //end for_each_immediate_neighbor

	if (greatest_temp_val != self_temp_val) {
		change[0] = 1;
	}
}



__global__ void make_coords_to_ids_LUT_kernel(iptr src) {
	get_dims_ids_and_check_bounds
	int id = (row * src_cols) + col;
	src(row, col) = id;
}



void separate_blobs_launch(gMat& labels) {

	int N = labels.rows * labels.cols;

	dim3 num_blocks;
	dim3 threads_per_block;
	boilerplate->get_kernel_structure(labels, &num_blocks, &threads_per_block, 2, 2);

	cv::cuda::GpuMat temp_labels = labels;
	cv::cuda::GpuMat id_LUT(labels.size(), labels.type());
	make_coords_to_ids_LUT_kernel << < num_blocks, threads_per_block >> > (id_LUT);
	cudaDeviceSynchronize();

	int change = 0;
	int* h_flag = &change;
	int* d_flag;
	cudaMalloc(&d_flag, sizeof(int));
	bool converged = false;

	while (!converged) {
		cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);
		linear_flow_kernel <<<num_blocks, threads_per_block>>> (labels, temp_labels, id_LUT, N, flag);
		cudaDeviceSynchronize();
		cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

		if (change == 0) {
			converged = true;
		}
		change = 0;
	}
	cudaFree(d_flag);

	labels = temp_labels;
}