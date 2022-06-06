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
	alias_input(labels);
	get_structure_from_mat;

	make_2d_kernel_from_structure;
	cv::cuda::GpuMat flags(cv::Size(N, 1), labels.type(), cv::Scalar(0));
	raise_flags_kernel<<<num_blocks, threads_per_block>>>(labels, flags);
	cusyncerr(raise_falgs_in_produce_ordered_labels);

	make_1d_kernel_from_structure;
	init_map_kernel<<<num_blocks, threads_per_block>>>(flags);
	cusyncerr(init_map_in_produce_ordered_labels);

	int K = 0;
	cv::cuda::GpuMat temp_map;
	pop_elm_vec_launch(flags, temp_map, 0, &K);

	cv::cuda::GpuMat inverted_map(flags.size(), flags.type(), cv::Scalar(0));
	invert_map_kernel<<<num_blocks, threads_per_block >>>(temp_map, inverted_map);
	cusyncerr(invert_map_in_produce_ordered_labels);

	make_2d_kernel_from_structure;
	assign_new_labels_kernel<<<num_blocks, threads_per_block >>>(labels, inverted_map);
	cusyncerr(assign_new_labels_in_produce_ordered_labels);

	*num_labels = K;
}