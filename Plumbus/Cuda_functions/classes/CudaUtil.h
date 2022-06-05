#pragma once
#include"../../cuda_includes.h"
//could be an issue here, cuda util might be including itself were it not for the pragma once





class CudaUtil {
public:
	CudaUtil();
	~CudaUtil();


	void get_kernel_structure(gMat& input, dim3* num_blocks, dim3* threads_per_block, int mat_dimensionality, int kernel_dimensionality);












private:



};