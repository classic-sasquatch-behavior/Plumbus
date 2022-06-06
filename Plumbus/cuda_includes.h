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

#include"Cuda_functions/classes/cuda_macros.h"




