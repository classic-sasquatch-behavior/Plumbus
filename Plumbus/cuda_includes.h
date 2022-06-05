#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/core/cuda.hpp>
#include <opencv2/cudev/ptr2d/glob.hpp>

#include<device_launch_parameters.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/distance.h>

//std
#include"standard_includes.h"



typedef cv::cuda::PtrStepSzi iptr;
typedef cv::cuda::PtrStepSzf fptr;

typedef cv::cuda::GpuMat gMat;


#include"Cuda_functions/classes/CudaUtil.h"
//class CudaUtil;
extern CudaUtil* boilerplate;


//assumes that the exemplar mat is named src, returns row and col id. incidentally defines src_rows and src_cols
#define get_dims_ids_and_check_bounds																						\
	int src_rows = src.rows; int src_cols = src.cols;																		\
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; int col = (blockIdx.x * blockDim.x) + threadIdx.x;					\
	if (row >= src_rows || col >= src_cols) { return; }


//returns neighbor_row and neighbor_col as indices relative to row and col of thread
#define for_each_immediate_neighbor(content_of_expression)																	\
	for (int irow = -1; irow <= 1; irow++) { for (int icol = -1; icol <= 1; icol++)											\
	{ int neighbor_row = row + irow; int neighbor_col = col + icol;															\
	if (neighbor_row >= src_rows || neighbor_col >= src_cols || neighbor_row < 0 || neighbor_col < 0) { break; }			\
	content_of_expression																									\
	}}


//simple as. leave out the semicolon and add it in the code for formatting reasons.
#define cusync 	cudaDeviceSynchronize()





