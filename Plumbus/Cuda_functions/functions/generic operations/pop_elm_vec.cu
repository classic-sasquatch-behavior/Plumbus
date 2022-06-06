#include"../../../cuda_includes.h"
#include"../../../config.h"
#include"../../headers/generic operations/exclusive_scan_vec.cuh"






__global__ void raise_flags(iptr src, iptr dst, int pop_elm) {
	get_dims_ids_and_check_bounds

	int val = src(0, id);
	if (val == pop_elm) { return; }
	dst(0, id) = 1;
}



__global__ void construct_popped_vector(iptr src, iptr sums, iptr dst, int pop_elm) {
	get_dims_ids_and_check_bounds

	int val = src(0, id);
	if (val == pop_elm) { return; }
	int index = sums(0, id);
	dst(0, index) = val;
}










void pop_elm_vec_launch(gMat& input, gMat& output, int pop_elm, int* max = 0) {

	//vector is assumed to be 1d and horizontal

	dim3 num_blocks;
	dim3 threads_per_block;
	boilerplate->get_kernel_structure(input, &num_blocks, &threads_per_block, 1, 1);

	cv::cuda::GpuMat flags(input.size(), input.type(), cv::Scalar(0));
	raise_flags << <num_blocks, threads_per_block >> > (input, flags, pop_elm);
	cusyncerr(raise_falgs_in_pop_elm_vec);

	int K = 0;
	cv::cuda::GpuMat scan_result(input.size(), input.type());
	exclusive_scan_vec_launch(flags, scan_result, &K);

	cv::cuda::GpuMat dest(cv::Size(K, 1), input.type());
	construct_popped_vector <<<num_blocks, threads_per_block >>> (input, scan_result, dest, pop_elm);
	cusyncerr(construct_popped_vector_in_pop_elm_vec);

	output = dest;
	max = &K;
}