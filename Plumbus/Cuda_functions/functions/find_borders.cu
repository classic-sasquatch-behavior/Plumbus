#include"../../cuda_includes.h"








__global__ void find_borders_kernel(int width, int height, cv::cuda::PtrStepi d_src_in,  int* d_focus_out, int* d_target_out) { 

	int focus_x = blockIdx.x;
	int focus_y = blockIdx.y;

	//printf("input rows: %s, input cols: %s", d_src_in.rows, d_src_in.cols);

	int target_x_mod = (threadIdx.x % 3) - 1;
	int target_y_mod = ((threadIdx.x - target_x_mod) / 3) - 1;

	int target_x = focus_x + target_x_mod;
	int target_y = focus_y + target_y_mod;

	if (focus_x >= width || focus_y >= height || target_x >= width || target_y >= height || target_x < 0 || target_y < 0) { return; }

	int write_position = (((blockIdx.x + (gridDim.x * blockIdx.y)) * 9) + threadIdx.x);

	int focus_val = d_src_in(focus_y, focus_x);
	int target_val = d_src_in(target_y, target_x);
	//printf("focus val: %s, target val: %s \n", focus_val, target_val);

	if (focus_val != target_val) {
		d_focus_out[write_position] = focus_val;
		d_target_out[write_position] = target_val;
	}
	else {
		d_focus_out[write_position] = -1;
		d_target_out[write_position] = -1;
	}
}






struct make_pair : public thrust::binary_function<int, int, thrust::pair<int, int>> {
	__host__ __device__
		thrust::pair<int, int> operator()(int x, int y) { return thrust::make_pair(x, y); }
};


std::vector<thrust::pair<int, int>> find_borders_launch(int src_width, int src_height, cv::Mat d_src) { //just make it return a thrust::host_vector

	thrust::device_vector<int> d_focus_result(src_width * src_height * 9);
	thrust::device_vector<int> d_target_result(src_width * src_height * 9);

	int* d_focus = thrust::raw_pointer_cast(d_focus_result.data());
	int* d_target = thrust::raw_pointer_cast(d_target_result.data());

	dim3 num_blocks = {uint(src_width + 1), uint(src_height + 1)};
	dim3 threads_per_block = {9, 1, 1};

	cv::cuda::GpuMat input(cv::Size(src_width, src_height), CV_32SC1);

	input.upload(d_src);

	find_borders_kernel <<<num_blocks, threads_per_block>>> (src_width, src_height, input, d_focus, d_target);

	// make the host block until the device is finished with foo
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("\nCUDA error in find_borders_kernel: %s\n", cudaGetErrorString(error));
		exit(-1);
	}



	int unique_length = d_focus_result.size();

	thrust::device_vector<thrust::pair<int, int>> d_unique(unique_length); //length doesn't matter. fails at any length.

	thrust::transform(d_focus_result.begin(), d_focus_result.end(), d_target_result.begin(), d_unique.begin(), make_pair() ); //seems to do its job just fine

	thrust::sort(thrust::device, d_unique.begin(), d_unique.end());

	thrust::device_vector<thrust::pair<int, int>>::iterator new_end  = thrust::unique(thrust::device, d_unique.begin(), d_unique.end());

	int new_length = thrust::distance(d_unique.begin(), new_end);

	std::vector<thrust::pair<int, int>> output(new_length);

	thrust::copy(d_unique.begin(), d_unique.begin() + new_length, output.begin() );

	return output;
}