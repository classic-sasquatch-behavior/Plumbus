#include"../../../cuda_includes.h"
#include"../../../config.h"







__global__ void exclusive_scan_vec_kernel(iptr source, iptr result, int N, int* max) {











}








void exclusive_scan_vec_launch(gMat& input, gMat& output, int* sum) {

	int true_length = input.cols;
	int closest_greater_power_of_2;

	//find closest power of 2

	int padding = closest_greater_power_of_2 - true_length;
	int N = true_length + padding;

	


	


	unsigned int;













}