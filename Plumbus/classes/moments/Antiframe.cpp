#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"

#pragma region get-set
void Antiframe::set_past(Moment* input) {
	_past = dynamic_cast<Frame*>(input);
}

void Antiframe::set_future(Moment* input) {
	_future = dynamic_cast<Frame*>(input);
}

#pragma endregion

#pragma region init

Antiframe::Antiframe(int antiframe_index, Clip* parent) : Moment() {
	_parent = parent;
	_antiframe_index = antiframe_index;

}

void Antiframe::run_filters() {
	calculate_layer1();
	quantify_difference();
}

#pragma endregion

#pragma region body

void Antiframe::calculate_layer1() { //make faster by putting on gpu
	cv::Mat first = _past->source();
	cv::Mat second = _future->source();

	cv::Mat* result = new cv::Mat(first.size(), CV_8UC1);

	for (int row = 0; row < first.rows; row++) {
		for (int col = 0; col < first.cols; col++) {
			cv::Vec3b first_pixel = first.at<cv::Vec3b>(row, col);
			cv::Vec3b second_pixel = second.at<cv::Vec3b>(row, col);

			int difference_sum = 0;
			for (int channel = 0; channel < 3; channel++) {
				difference_sum += abs(first_pixel[channel] - second_pixel[channel]);
			}
			uchar color;
			if (difference_sum > 10) {
				color = 255;
			}
			else {
				color = 0;
			}
			result->at<uchar>(row, col) = color;
		}
	}
	_layer1 = *result;
}

void Antiframe::quantify_difference() {
	float difference_ratio = 0;
	cv::Size histSize = { 256, 1 };
	cv::cuda::GpuMat hist(histSize, CV_32SC1);
	cv::Mat plainhist(histSize, CV_32SC1);
	cv::cuda::GpuMat target;
	target.upload(layer1());

	cv::cuda::calcHist(target, hist);

	hist.download(plainhist);
	
	float num_pixels = layer1().rows * layer1().cols;

	//stable for large videos but not general enough for smaller ones (it still picks up hard cuts, though)
	float num_black = (plainhist.at<int>(0) + 1) / num_pixels;
	float num_white = plainhist.at<int>(255) / num_pixels;

	//more general but potentially unstable. this is a rabbithole. Quantize the parameters/metrics to 30fps and make calculations from there.
	//float num_black = (plainhist.at<int>(0) + 1);
	//float num_white = plainhist.at<int>(255);

	difference_ratio = (num_white / num_black);
	_difference_ratio = difference_ratio;
	parent()->add_difference_value(difference_ratio);
}

#pragma endregion




