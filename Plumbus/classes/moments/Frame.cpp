#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"
//begin frame

#pragma region structors

Frame::Frame(cv::Mat source, int frame_index, Clip* parent) : Moment() {
	_source = source;
	_frame_index = frame_index;
	_parent = parent;
}

Frame::~Frame() {

}

#pragma endregion

#pragma region get-set
cv::Mat Frame::past_difference() { return _past->layer1(); }
cv::Mat Frame::future_difference() { return _future->layer1(); }

void Frame::set_past(Moment* input) {
	_past = dynamic_cast<Antiframe*>(input);
}

void Frame::set_future(Moment* input) {
	_future = dynamic_cast<Antiframe*>(input);
}

#pragma endregion

#pragma region setup

void Frame::run_filters() {

	//timer->begin("selective blur");
	_blurred = GPU->selective_blur(source(), 50, 10, 5);
	//timer->end("selective blur");

	//timer->begin("generate superpixels");
	generate_superpixels(blurred());
	//timer->end("generate superpixels");

	find_regions();

	//timer->begin("draw superpixels");
	_superpixels = draw_superpixels(field());
	//timer->end("draw superpixels");

	//timer->begin("find edges");
	_edges = find_edges(blurred());
	//timer->end("find edges");

	//timer->begin("overlay edges");
	_edge_overlay = binary_overlay(superpixels(), edges());
	//timer->end("overlay edges");

	_regions = draw_regions(field());
}

void Frame::identify_local_objects() {

}

void Frame::generate_superpixels(cv::Mat input) {
	cv::Mat src_HSV;
	cv::cvtColor(input, src_HSV, cv::COLOR_BGR2HSV);
	const int NUM_ITERATIONS = 10; //original: 10 
	const int MIN_ELEMENT_SIZE = 25; //original: 25 
	const int REGION_SIZE = 10; //original: 10 
	const float RATIO = 0.075; //original: 0.075  

	timer->begin("opencv superpixel function"); //EXTREMELY SLOW //try other opencv implimenations (3-4 others), then see if you can fix the github one, then try to build your own
	cv::Ptr<cv::ximgproc::SuperpixelLSC> superpixels = cv::ximgproc::createSuperpixelLSC(src_HSV, REGION_SIZE, RATIO);

	//timer->begin("cv iterate");
	superpixels->iterate(NUM_ITERATIONS);
	//timer->end("cv iterate");

	if (MIN_ELEMENT_SIZE > 0) {
		superpixels->enforceLabelConnectivity(MIN_ELEMENT_SIZE);
	}
	timer->end("opencv superpixel function"); //EXTREMELY SLOW


	//FAST
	int num_superpixels = superpixels->getNumberOfSuperpixels();
	cv::Mat labels;
	superpixels->getLabels(labels);
	cv::Mat src = source();
	Field* new_field = new Field(this, labels);
	set_field(new_field);
	//FAST


	//FAST
	for (int superpixel = 0; superpixel < num_superpixels; superpixel++) {
		Superpixel* new_superpixel = new Superpixel(new_field);
		new_field->add_superpixel(new_superpixel);
	}
	//FAST


	//timer->begin("assign points to superpixels"); //A LITTLE SLOW BUT FAST ENOUGH
	for (int row = 0; row < labels.rows; row++) {
		for (int col = 0; col < labels.cols; col++) {
			cv::Point new_point(col, row);
			int label = labels.at<int>(new_point);
			cv::Vec3b new_color = input.at<cv::Vec3b>(new_point);
			field()->superpixel_at(label)->add_point(new_point);
			field()->superpixel_at(label)->add_color(new_color);
		}
	}
	//timer->end("assign points to superpixels"); //A LITTLE SLOW BUT FAST ENOUGH


	timer->begin("wake up superpixels"); //A LITTLE SLOW BUT FAST ENOUGH
	for (Superpixel* superpixel : field()->all_superpixels()) {
		superpixel->compute_average_color();
		superpixel->compute_mean_of_points();
		superpixel->compute_histogram();
	}
	timer->end("wake up superpixels"); //A LITTLE SLOW BUT FAST ENOUGH

	//timer->begin("initialize bins");
	field()->initialize_bins();
	//timer->end("initialize bins");

	timer->begin("connect neighbors");
	field()->connect_neighbors();
	timer->end("connect neighbors");


}

void Frame::find_regions() {
	//timer->begin("form regions");
	field()->form_regions();
	//timer->end("form regions");

	timer->begin("connect regions");
	field()->connect_regions(); //VERY SLOW
	timer->end("connect regions");

	timer->begin("calculate average region colors");
	field()->calculate_average_region_colors();
	timer->end("calculate average region colors");



	timer->begin("refine regions sequence"); //quite fast for what it is
	//consider making regions know their region neighbors
	field()->refine_region_sequence_naive();
	//field()->refine_region_sequence();
	timer->end("refine regions sequence");


}

#pragma endregion

#pragma region filters

cv::Mat Frame::draw_regions(Field* input) {
	cv::Mat output(source().size(), source().type());

	for (Region* region : input->all_regions()) {
		for (Superpixel* constituent : region->all_constituents()) {
			for (cv::Point point : constituent->all_points()) {
				output.at<cv::Vec3b>(point) = region->average_color();
			}
		}
	}








	return output;
}

cv::Mat Frame::draw_superpixels(Field* input) {
	int num_superpixels = input->num_superpixels();
	cv::Mat output(source().size(), source().type());

	for (int superpixel = 0; superpixel < num_superpixels; superpixel++) {
		Superpixel* this_pixel = input->superpixel_at(superpixel);
		cv::Vec3b color_to_set = this_pixel->average_color();
		for (int pixel = 0; pixel < this_pixel->num_points(); pixel++ ) {
			output.at<cv::Vec3b>(this_pixel->point_at(pixel)) = color_to_set;
		}
	}
	return output;
}

cv::Mat Frame::binary_overlay(cv::Mat base_in, cv::Mat mask_in) {
	cv::Mat base, mask;
	cv::cvtColor(mask_in, mask, cv::COLOR_GRAY2BGR);
	base_in.copyTo(base);
	base += mask;
	return base;
}

cv::Mat Frame::find_edges(cv::Mat input) {
	float ratio = 2;
	double thresh_1 = 80;
	double thresh_2 = thresh_1 * ratio;
	int aperture_size = 3;
	cv::Mat output;
	cv::Canny(input, output, thresh_1, thresh_2, aperture_size );
	return output;
}

cv::Mat Frame::selective_blur_hsv(cv::Mat input) {

	cv::Mat hsv;
	cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

	int ITERATIONS = 20;
	int H_THRESH = 2;
	int SV_THRESH = 10;

	//hsv: 0-179, 0-255, 0-255

	for (int i = 0; i < ITERATIONS; i++) {
		cv::Mat temp = hsv;
		std::cout << i << std::endl;
		for (int row = 0; row < input.rows; row++) {
			for (int col = 0; col < input.cols; col++) {
				cv::Vec3b focus_color = hsv.at<cv::Vec3b>(row, col);
				int num_neighbors = 0;
				int H_sum = 0;
				int S_sum = 0;
				int V_sum = 0;

				for (int irow = -1; irow <= 1; irow++) {
					for (int icol = -1; icol <= 1; icol++) {
						int target_row = row + irow;
						int target_col = col + icol;

						if ((target_row > -1)&&(target_col > -1)&&(target_row < hsv.rows)&&(target_col < hsv.cols)) {
							cv::Vec3b target_color = hsv.at<cv::Vec3b>(target_row, target_col);
							int H_difference = 0;
							int SV_difference = 0;



							for (int channel = 0; channel < 3; channel++) {

								switch ( channel == 0) {
								case true:
									H_difference += abs(focus_color[channel] - target_color[channel]);
									break;
								case false:
									SV_difference += abs(focus_color[channel] - target_color[channel]);
									break;
								}

								if ((H_difference <= H_THRESH)&&(SV_difference <= SV_THRESH)) {
									num_neighbors++;
									H_sum += target_color[0];
									S_sum += target_color[1];
									V_sum += target_color[2];
								}
							}
						}
					}
				}
				focus_color[0] = H_sum / num_neighbors;
				focus_color[1] = S_sum / num_neighbors;
				focus_color[2] = V_sum / num_neighbors;

				temp.at<cv::Vec3b>(row, col) = focus_color;
			}
		}
		hsv = temp;
	}
	cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

	return hsv;
}

cv::Mat Frame::custom_segmentation(cv::Mat input) {


	return input;
}

cv::Mat Frame::contour_segmentation(cv::Mat input){

	cv::Mat gray;
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

	cv::Mat thresh;
	cv::threshold(gray, thresh, 150, 255, cv::THRESH_BINARY);

	return thresh;
}

cv::Mat Frame::watershed_segmentation(cv::Mat input) {

	//create kernel to sharpen image - approximation of second derivative
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
										1,  1, 1,
										1, -8, 1,
										1,  1, 1);

	//sharpen
	cv::Mat imgLaplacian;
	cv::filter2D(input, imgLaplacian, CV_32F, kernel);
	cv::Mat sharp;
	input.convertTo(sharp, CV_32F);
	cv::Mat imgResult = sharp - imgLaplacian;

	//convert back to 8 bits grayscale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

	//create binary image from source image
	cv::Mat bw;
	cv::cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
	cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);



	////perform the distance transform algorithm
	//cv::Mat dist;
	//cv::distanceTransform(bw, dist, cv::DIST_L2, 3); //original: DIST_L2, 3

	////normalize the distance image
	//cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

	////threshold to obtain the peaks, which will be markers for foreground objects
	//cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);

	////dilate the dist image
	//cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
	//cv::dilate(dist, dist, kernel1);


	//create the CV-8U version of dist, needed to find contours
	cv::Mat dist_8u;
	bw.convertTo(dist_8u, CV_8U);

	//find total markers
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//create the marker image for the watershed algorithm
	cv::Mat markers = cv::Mat::zeros(bw.size(), CV_32S);

	//draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++) {
		cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
	}



	//draw the background marker
	//cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255),-1);
	cv::Mat markers8U;
	markers.convertTo(markers8U, CV_8U, 10);
	return markers8U;

	//perform the watershed algorithm
	cv::watershed(imgResult, markers);

	//unexplained
	cv::Mat mark;
	markers.convertTo(mark, CV_8U);
	cv::bitwise_not(mark, mark);

	//get colors
	std::vector<cv::Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		colors.push_back(colorwheel->get_color(64));
	}

	//create the result image
	cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);

	//fill labeled objects with random colors
	for (int row = 0; row < markers.rows; row++) {
		for (int col = 0; col < markers.cols; col++) {

			int index = markers.at<int>(row, col);
			if (index > 0 && index <= static_cast<int>(contours.size())) {
				dst.at<cv::Vec3b>(row, col) = colors[index - 1];
			}

		}
	}

	return dst;
}

cv::Mat Frame::kmeans_segmentation(cv::Mat input) {
	const int K = 8;
	
	cv::Mat blur = input;
	//cv::blur(blur, blur, cv::Size(3, 3));

	cv::Mat data;
	blur.convertTo(data, CV_32F);
	data = data.reshape(1, data.total());

	cv::Mat labels, centers;
	cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

	centers = centers.reshape(3, centers.rows);
	data = data.reshape(3, data.rows);

	cv::Vec3f* p = data.ptr<cv::Vec3f>();
	for (int i = 0; i < data.rows; i++) {
		int center_id = labels.at<int>(i);
		p[i] = centers.at<cv::Vec3f>(center_id);
	}

	data = data.reshape(3, input.rows);
	data.convertTo(data, CV_8U);


	return data;
}

cv::Mat Frame::split_segmentation(cv::Mat input) {
	std::set<std::vector<int>> unique_colors;
	
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {

			cv::Vec3b color = input.at<cv::Vec3b>(row, col);
			unique_colors.insert({color[0], color[1], color[2]});
		}
	}

	std::vector<cv::Mat> color_layers;
	for (std::vector<int> color : unique_colors) {
		cv::Mat color_layer;
		cv::Scalar lower_bound = {double(color[0]), double(color[1]), double(color[2])};
		cv::Scalar upper_bound = lower_bound;
		cv::inRange(input, lower_bound, upper_bound, color_layer);
		color_layers.push_back(color_layer);
	}

	uchar black = 0;
	uchar white = 255;
	cv::Mat output(input.size(), input.type());

	for (cv::Mat layer : color_layers) {
		cv::Vec3b color = colorwheel->get_color(256);
		for (int row = 0; row < output.rows; row++) {
			for (int col = 0; col < output.cols; col++) {
				if (layer.at<uchar>(row, col) == white) {
					output.at<cv::Vec3b>(row, col) = color;
				}
			}
		}
	}

	return output;










	//calculate histogram to obtain the number of colors

	//use inrange to create a mask of each color

	//separate elements in masks by their contiguity











	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {


			






		}
	}








}

#pragma endregion


