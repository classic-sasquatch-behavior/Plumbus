#pragma once
#include"../../includes.h"


class Util {
public:
	Util();
	~Util();

	void print_mat(cv::Mat input, int depth = 0);
	void print_gpu_mat(cv::cuda::GpuMat input, int depth);








private:

};