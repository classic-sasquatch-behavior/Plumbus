#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"



Superpixel::Superpixel(Field* parent) {
	_parent = parent;
}

Superpixel::~Superpixel() {

}



void Superpixel::compute_histogram() {
	cv::Mat B_hist(cv::Size(256, 1), CV_32FC1, cv::Scalar{ 0 });
	cv::Mat G_hist(cv::Size(256, 1), CV_32FC1, cv::Scalar{ 0 });
	cv::Mat R_hist(cv::Size(256, 1), CV_32FC1, cv::Scalar{ 0 });

	std::vector<cv::Mat> hist_channels = { B_hist, G_hist, R_hist };

	for (cv::Vec3b color : all_colors()) {
		for (int channel = 0; channel < 3; channel++) {
			hist_channels[channel].at<float>(color[channel])++;
		}
	}

	float norm_factor = float(num_points()) / 100.0f;

	for (int channel = 0; channel < 3; channel++) {

		for (int i = 0; i < 256; i++) {
			hist_channels[channel].at<float>(i) /= norm_factor;

		}
	}

	cv::Mat hist_out;
	cv::vconcat(hist_channels, 3, hist_out);

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
		cv::Vec3b source_color = color_at(i);
		for (int it = 0; it < 3; it++) {
			color_sum[it] += source_color[it];
		}
	}

	for (int i = 0; i < 3; i++) {
		color_sum[i] /= num_points();
	}

	_average_color[0] = color_sum[0];
	_average_color[1] = color_sum[1];
	_average_color[2] = color_sum[2];
}