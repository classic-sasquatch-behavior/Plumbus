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
