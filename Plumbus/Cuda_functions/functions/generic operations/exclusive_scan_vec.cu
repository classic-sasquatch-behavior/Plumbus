#include"../../../cuda_includes.h"
#include"../../../config.h"







__global__ void upsweep(iptr src, iptr buff, int offset, int step, int* max) {
	get_dims_ids_without_bounds
	if (id >= step) { return; }
	int ai = offset * (2 * id + 1) - 1;
	int bi = offset * (2 * id + 2) - 1;
	buff(0, bi) += src(0, ai);

	if (step == 1) {
		if (id == 0) {
			*max = buff(0, src_cols - 1); //probably worng- wondering if I want to take the final sum from buff or src
			buff(0, src_cols - 1) = 0;
		}
	}

}



__global__ void downsweep(iptr src, iptr buff, int offset, int step) {
	get_dims_ids_without_bounds
	if (id >= step) { return; }
	int ai = offset * (2 * id + 1) - 1;
	int bi = offset * (2 * id + 2) - 1;

	float t = src[ai];
	buff[ai] = src[bi];
	buff[bi] += t;
}


__global__ void truncate_zeros(iptr src, iptr dst, int N) {
	get_dims_ids_without_bounds
	if (id >= N) { return; }
	dst(0, id) = src(0, id); //if youre not careful, this will be keeping the zeros and chooping off the leading numbers

}


__global__ void pad_input(iptr src, iptr dst) {
	get_dims_ids_and_check_bounds
	dst(0, id) = src(0, id);
}







void exclusive_scan_vec_launch(gMat& input, gMat& output, int* sum) {

	int true_length = input.cols;
	int closest_greater_power_of_2 = pow(2, ceil(log(true_length) / log(2)));
	int padding = closest_greater_power_of_2 - true_length;
	int N = true_length + padding;	//actual length that we're gonna pass into the kernel

	dim3 num_blocks;
	dim3 threads_per_block;
	cv::cuda::GpuMat padded_input(cv::Size(N, 1), input.type(), cv::Scalar(0));
	boilerplate->get_kernel_structure(input, &num_blocks, &threads_per_block,1, 1);
	pad_input << <num_blocks, threads_per_block >> > (input, padded_input);
	cusyncerr(pad_input_in_exclusive_scan_vec);

	boilerplate->get_kernel_structure(padded_input, &num_blocks, &threads_per_block, 1, 1);
	cv::cuda::GpuMat padded_output = padded_input;
	int max = 0;
	int* h_max = &max;
	int* d_max;
	cudaMalloc(&d_max, sizeof(int));
	


	int offset = 1;

	for (int step = N >> 1; step > 0; step>>=1) {
		padded_output = padded_input;
		upsweep<<<num_blocks, threads_per_block>>>(padded_input, padded_output, offset, step, d_max);
		cusyncerr(upsweep_in_exclusive_scan_vec);
		padded_input = padded_output;
		offset *= 2;
	}
	cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_max);

	for (int step = 1; step < N; step *=2) {
		offset >>= 1;
		padded_output = padded_input;
		downsweep<<<num_blocks, threads_per_block>>>(padded_input, padded_output, offset, step);
		cusyncerr(downsweep_in_exclusive_scan_vec);
		padded_input = padded_output;
	}

	truncate_zeros <<<num_blocks, threads_per_block>>> (padded_input, output, true_length);
	cusyncerr(truncate_zeros_in_exclusive_scan_vec);

	sum = &max;
}