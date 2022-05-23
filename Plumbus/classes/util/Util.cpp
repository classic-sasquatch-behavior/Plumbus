#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"







Util::Util() {

}


void Util::print_mat(cv::Mat input, int depth) {
	int working_depth = depth;
	cv::Mat working(input.size(), CV_32SC1);
	input.convertTo(working, CV_32SC1);
	if (working_depth == 0) { working_depth = input.rows; }

	std::cout << std::endl;
	for (int row = 0; row < working_depth; row++) {
		std::cout << "|";
		for (int col = 0; col < working_depth; col++) {
			std::cout << working.at<int>(row, col) << ", ";
		}
		std::cout << "|" << std::endl;
	}
	std::cout << std::endl;
}


void Util::print_gpu_mat(cv::cuda::GpuMat input, int depth) {
	cv::Mat temp(input.size(), input.type());
	input.download(temp);
	temp.convertTo(temp, CV_32SC1);

	int new_start_row = temp.rows - depth;
	int new_end_row = temp.rows;
	int new_start_col = temp.cols - depth;
	int new_end_col = temp.cols;


	for (int row = 0; row < depth; row++) {
		for (int col = 0; col < depth; col++) {
			std::cout << temp.at<int>(row, col) << ", ";
		}
		std::cout << " ... ";

		for (int col = new_start_col; col < new_end_col; col++) {
			std::cout << temp.at<int>(row, col) << ", ";
		}
		std::cout << "|" << std::endl << std::endl;
	}
	std::cout << "..." << std::endl << std::endl;
	for (int row = new_start_row; row < new_end_row; row++) {
		for (int col = 0; col < depth; col++) {
			std::cout << temp.at<int>(row, col) << ", ";
		}
		std::cout << " ... ";
		for (int col = new_start_col; col < new_end_col; col++) {
			std::cout << temp.at<int>(row, col) << ", ";
		}
		std::cout << "|" << std::endl << std::endl;
	}
	std::cout << std::endl;
}