#include"../../../cuda_includes.h"






__global__ void update_centers_kernel(cv::cuda::PtrStepSzi labels, cv::cuda::PtrStepSzi row_vals, cv::cuda::PtrStepSzi col_vals) {



	if () { return; }



}




void update_centers_launch(cv::cuda::GpuMat& labels, cv::cuda::GpuMat& row_vals, cv::cuda::GpuMat& col_vals) {

	int center_rows = row_vals.rows;
	int center_cols = row_vals.cols;
	int num_centers = center_rows * center_cols;
	

	dim3 num_blocks();
	dim3 threads_per_block();



	update_centers_kernel << <num_blocks, threads_per_block >> > (labels, row_vals, col_vals);




}