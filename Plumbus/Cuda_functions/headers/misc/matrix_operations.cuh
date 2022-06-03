#pragma once
#include"../../../cuda_includes.h"


//probably would be faster to just return a pointer, or to end up making this set of functions inline













void add_launch(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output);

void subtract_launch(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output);

void multiply_launch(cv::Mat &input_a, cv::Mat &input_b, cv::Mat &output);

//template<typename AnyType>
//void sum_launch(cv::Mat &input_a, cv::Mat &input_b, AnyType &output) {
//	
//}
//

