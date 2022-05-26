#include"../../../cuda_includes.h"





__global__ void add_kernel(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf Out) {

	int x = (blockDim.x * blockIdx.x) + threadIdx.x ;
	int y = threadIdx.y;

	int input_width = A.cols;
	int input_height = A.rows;
	if (x > input_width || y > input_height) { return; }

	Out(x, y) = A(x, y) + B(x, y);
}

__global__ void subtract_kernel(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf Out) {
	int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int y = threadIdx.y;

	int input_width = A.cols;
	int input_height = A.rows;
	if (x > input_width || y > input_height) { return; }

	Out(x, y) = A(x, y) - B(x, y);
}

__global__ void multiply_kernel(cv::cuda::PtrStepSzf A, cv::cuda::PtrStepSzf B, cv::cuda::PtrStepSzf Out) {

	int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int y = (blockDim.y * blockIdx.y) + threadIdx.y;

	int input_width = A.cols;
	int input_height = A.rows;
	if (x > input_width || y > input_height) { return; }

	Out(x, y) = A(x, y) * B(x, y);
}


//do this last, it's gonna be complicated as fuck
//template<typename AnyType>
//__global__ void sum_kernel() {
//	
//
//
//
//
//
//}





//TODO: finish installing CUDA intellisense 

void add_launch(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output) {
	cv::cuda::GpuMat d_input_a;
	cv::cuda::GpuMat d_input_b;
	cv::cuda::GpuMat d_output(input_b.size(), input_b.type());

	d_input_a.upload(input_a);
	d_input_b.upload(input_b);

	dim3 num_blocks = { (unsigned int)(((input_a.cols - (input_a.rows % 32)) / 32) + 1), (unsigned int)(((input_a.rows - (input_a.rows % 32)) / 32)) + 1, 1 };
	dim3 threads_per_block = { 32, (unsigned int)((input_a.rows % 32) + 1), 1 };

	add_kernel <<<num_blocks, threads_per_block>>>(d_input_a, d_input_b, d_output);

	d_output.download(output);

}

void subtract_launch(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output) {
	cv::cuda::GpuMat d_input_a;
	cv::cuda::GpuMat d_input_b;
	cv::cuda::GpuMat d_output(input_b.size(), input_b.type());

	d_input_a.upload(input_a);
	d_input_b.upload(input_b);

	dim3 num_blocks = { (unsigned int)(((input_a.cols - (input_a.rows % 32)) / 32) + 1), (unsigned int)(((input_a.rows - (input_a.rows % 32)) / 32)) + 1, 1 };
	dim3 threads_per_block = { 32, (unsigned int)((input_a.rows % 32) + 1), 1 };

	subtract_kernel <<<num_blocks, threads_per_block >>>(d_input_a, d_input_b, d_output);

	d_output.download(output);

}

void multiply_launch(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output) {
	cv::cuda::GpuMat d_input_a;
	cv::cuda::GpuMat d_input_b;
	cv::cuda::GpuMat d_output(input_b.size(), input_b.type());

	d_input_a.upload(input_a);
	d_input_b.upload(input_b);

	dim3 num_blocks = { (unsigned int)(((input_a.cols - (input_a.rows % 32)) / 32) + 1), (unsigned int)(((input_a.rows - (input_a.rows % 32)) / 32)) + 1, 1 };
	dim3 threads_per_block = { 32, (unsigned int)((input_a.rows % 32) + 1), 1 };

	multiply_kernel<<<num_blocks, threads_per_block >>>(d_input_a, d_input_b, d_output);

	d_output.download(output);

}

