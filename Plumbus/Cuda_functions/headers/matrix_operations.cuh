#pragma once
#include"../../cuda_includes.h"


//probably would be faster to just return a pointer, or to end up making this set of functions inline













cv::Mat add_launch(cv::Mat input_a, cv::Mat input_b);

cv::Mat subtract_launch(cv::Mat input_a, cv::Mat input_b);

cv::Mat multiply_launch(cv::Mat input_a, cv::Mat input_b);

template<typename AnyType>
AnyType sum_launch(cv::Mat input_a, cv::Mat input_b) {
	AnyType result;



}


