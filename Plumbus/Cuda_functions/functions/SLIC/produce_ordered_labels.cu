#include"../../../cuda_includes.h"
#include"../../../config.h"
#include"../../headers/generic operations/exclusive_scan_vec.cuh"
#include"../../headers/generic operations/pop_elm_vec.cuh"



__global__ void raise_flags_kernel(iptr src, iptr flags) {
	get_dims_ids_and_check_bounds
	int label_value = src(row, col);
	if (label_value == 0) { return; }
	flags(0, label_value) = 1;
}

//sum flags

__global__ void init_map_kernel(iptr src) {
	get_dims_ids_and_check_bounds
	int flag = src(0, id);
	src(0, id) = id * flag;
}

//pop zeroes

__global__ void invert_map_kernel(iptr src, iptr dst) {
	get_dims_ids_and_check_bounds
	int original_index = src(0, id);
	dst(0, original_index) = id;
}

__global__ void assign_new_labels_kernel(iptr src, iptr map) {
	get_dims_ids_and_check_bounds
	int original_label = src(row, col);
	int new_label = map(0, original_label);
	src(row, col) = new_label;
}






void produce_ordered_labels_launch(gMat& labels, int* num_labels) {
	int N = labels.rows * labels.cols;
	dim3 num_blocks_2d;
	dim3 threads_per_block_2d;
	boilerplate->get_kernel_structure(labels, &num_blocks_2d, &threads_per_block_2d, 2, 2);

	dim3 num_blocks_1d;
	dim3 threads_per_block_1d;
	boilerplate->get_kernel_structure(labels, &num_blocks_1d, &threads_per_block_1d, 2, 1);


	cv::cuda::GpuMat flags(cv::Size(N, 1), labels.type(), cv::Scalar(0));
	raise_flags_kernel<<<num_blocks_2d, threads_per_block_2d >>>(labels, flags);
	cusyncerr(raise_falgs_in_produce_ordered_labels);

	init_map_kernel<<<num_blocks_1d, threads_per_block_1d >>>(flags);
	cusyncerr(init_map_in_produce_ordered_labels);

	int K = 0;
	cv::cuda::GpuMat temp_map;
	pop_elm_vec_launch(flags, temp_map, 0, &K);

	cv::cuda::GpuMat inverted_map(flags.size(), flags.type(), cv::Scalar(0));
	invert_map_kernel<<<num_blocks_1d, threads_per_block_1d >>>(temp_map, inverted_map);
	cusyncerr(invert_map_in_produce_ordered_labels);

	assign_new_labels_kernel<<<num_blocks_2d, threads_per_block_2d >>>(src, inverted_map);
	cusyncerr(assign_new_labels_in_produce_ordered_labels);

	*num_labels = K;
}