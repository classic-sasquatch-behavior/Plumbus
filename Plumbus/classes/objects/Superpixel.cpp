#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"



Superpixel::Superpixel(Field* parent) {
	_parent = parent;
}

Superpixel::~Superpixel() {

}



void Superpixel::compute_histogram() {
	//vertical matrices
	cv::Mat B_hist(cv::Size(1, 256), CV_32FC1, cv::Scalar{ 0 });
	cv::Mat G_hist(cv::Size(1, 256), CV_32FC1, cv::Scalar{ 0 });
	cv::Mat R_hist(cv::Size(1, 256), CV_32FC1, cv::Scalar{ 0 });

	cv::Mat hist_channels[3] = { B_hist, G_hist, R_hist };

	for (cv::Vec3b color : all_colors_BGR()) {
		for (int channel = 0; channel < 3; channel++) {
			hist_channels[channel].at<float>(color[channel])++;
		}
	}

	//turn three 1x256 matrices into one 3x256 matrix
	cv::Mat hist_out(cv::Size(3, 256), B_hist.type());
	cv::hconcat(hist_channels, 3, hist_out);

	_histogram = hist_out;
}

void Superpixel::compute_mean_of_points() {
	cv::Scalar processed_mean = cv::mean(all_points());
	_mean = { int(processed_mean[0]), int(processed_mean[1]) };
	parent()->add_mean(mean());
}


void Superpixel::compute_average_color() {

	std::vector<int> color_sum = {0,0,0};
	for (int i = 0; i < num_points(); i++) {
		cv::Vec3b source_color = color_at_BGR(i);
		for (int it = 0; it < 3; it++) {
			color_sum[it] += source_color[it];
		}
	}

	for (int i = 0; i < 3; i++) {
		color_sum[i] /= num_points();
	}

	_average_color_BGR[0] = color_sum[0];
	_average_color_BGR[1] = color_sum[1];
	_average_color_BGR[2] = color_sum[2];





	//calculate hsv values as well I guess

	color_sum = { 0,0,0 };
	for (int i = 0; i < num_points(); i++) {
		cv::Vec3b source_color = color_at_HSV(i);
		for (int it = 0; it < 3; it++) {
			color_sum[it] += source_color[it];
		}
	}

	for (int i = 0; i < 3; i++) {
		color_sum[i] /= num_points();
	}

	_average_color_HSV[0] = color_sum[0];
	_average_color_HSV[1] = color_sum[1];
	_average_color_HSV[2] = color_sum[2];
}