#include"../../includes.h"
#include"../../classes.h"

#pragma region parameters
//	config-A: starting configuration. produces a large number of objects, many small,
//but which largely obey the actual object outlines. there are some spills, but I think that
//those should be able to be fixed when we compare regions across time to neaten them and turn 
//them into objects. alomg those lines, we can assemble objects through the rule: regions that
//move together are usually part of the same object. 
 
//	config-B: there is no config-B yet, I just wanted to demonstrate to myself how the 
//formatting of this section is going to work for fututre configurations.

//SIMPLIFY
const int SIMPLE_STEPS = 50; //config-A: 50 //
const int SIMPLE_THRESHOLD = 20; //config-A: 20 //
const int SIMPLE_KERNEL_SIZE = 5; //config-A: 5 //
	const int SIMPLE_KERNEL_END = ((SIMPLE_KERNEL_SIZE - 1) / 2); 
	const int SIMPLE_KERNEL_BEGIN = -1 * SIMPLE_KERNEL_END;

//FIND CONNECTIONS
const int CONNECTION_THRESHOLD = 1; //config-A: 1 //

const int CANNY_THRESHOLD = 30; //config-A: 30 //
const int CANNY_LOW_THRESHOLD = 100; //config-A: 100 //
const int CANNY_RATIO = 3; //config-A: 3 //
const int CANNY_KERNEL_SIZE = 3; //config-A: 3 //
#pragma endregion

#pragma region constructors

Simplifier::Simplifier(Frame* parent, cv::Mat source) {
	_parent = parent;
	_source = source;

	std::cout << "simplifying images..." << std::endl;
	//_source = GPU->selective_blur(_source, SIMPLE_STEPS, SIMPLE_THRESHOLD, SIMPLE_KERNEL_SIZE);
	
	selective_blur();

	//std::cout << "finding regions..." << std::endl;
	//find_regions();

	//std::cout << "finding edges..." << std::endl;
	//find_edges();

	parent->set_simplifier(this);
	parent->set_simplification(_source);
	//parent->set_simple_edges(edges());
}

SimplePixel::SimplePixel(int row, int col, cv::Vec3b color) {
	_row = row;
	_col = col;
	_color = color;
}

SimpleRegion::SimpleRegion(){

}


#pragma endregion

#pragma region Simplifier

void Simplifier::selective_blur() {
	cv::Mat working_image = source();

	for (int step = 0; step < SIMPLE_STEPS; step++) {
		std::cout << "step " << step << std::endl;
		timer->begin("simplify step");
		cv::Mat intermediate_image(working_image.size(), working_image.type());

		for (int row = 0; row < working_image.rows; row++ ) {
			for (int col = 0; col < working_image.cols; col++) {
				cv::Vec3b self_color = working_image.at<cv::Vec3b>(row, col);
				int BGR_sum[3] = { 0,0,0 };
				int num_pixels = 0;

				for (int irow = SIMPLE_KERNEL_BEGIN; irow <= SIMPLE_KERNEL_END; irow++) {
					for (int icol = SIMPLE_KERNEL_BEGIN; icol <= SIMPLE_KERNEL_END; icol++) {
						int target_row = row + irow;
						int target_col = col + icol;

						if ((target_row > -1)&&(target_col > -1)&&(target_row < working_image.rows)&&(target_col < working_image.cols)) {
							cv::Vec3b target_color = working_image.at<cv::Vec3b>(target_row, target_col);
							int total_difference = 0;
								
							for (int channel = 0; channel < 3; channel++) {
								int target_channel_color = target_color[channel];
								int self_channel_color = self_color[channel];
								int difference = abs(target_channel_color - self_channel_color);
								total_difference += difference;
							}

							if (total_difference <= SIMPLE_THRESHOLD) {
								num_pixels++;
								BGR_sum[0] += target_color[0];
								BGR_sum[1] += target_color[1];
								BGR_sum[2] += target_color[2];
							}
						}
					} 
				} 
				self_color = {uchar(BGR_sum[0]/num_pixels), uchar(BGR_sum[1]/num_pixels), uchar(BGR_sum[2]/num_pixels)};
				intermediate_image.at<cv::Vec3b>(row, col) = self_color;
			}
		}
		working_image = intermediate_image;
		timer->end("simplify step");
	}
	set_source(working_image);
}

void Simplifier::find_regions() {

#pragma region initialize pixels
	std::cout << "initializing pixels..." << std::endl;
	timer->begin("initialize pixels");

	std::vector<std::vector<SimplePixel*>> new_pixels;

	for (int row = 0; row < source().rows; row++) {
		std::vector<SimplePixel*> new_row;
		for (int col = 0; col < source().cols; col++) {
			cv::Vec3b color = source().at<cv::Vec3b>(row, col);
			SimplePixel* new_pixel = new SimplePixel(row, col, color);
			new_row.push_back(new_pixel);
		}
		new_pixels.push_back(new_row);
	}
	set_pixels(new_pixels);

	timer->end("initialize pixels");
#pragma endregion initialize pixels

#pragma region find connections
	std::cout << "finding connections..." << std::endl;
	timer->begin("find connections");

	for (int row = 0; row < source().rows; row++) {
		for (int col = 0; col < source().cols; col++) {
			SimplePixel* focus = pixel(row, col);
			cv::Vec3b focus_color = focus->color();

			for (int irow = -1; irow <= 1; irow++) {
				for (int icol = -1; icol <= 1; icol++ ) {
					int target_row = row + irow;
					int target_col = col + icol;
					bool target_in_bounds = ((target_row > -1) && (target_col > -1) && (target_row < source().rows) && (target_col < source().cols));

					if (target_in_bounds) {
						SimplePixel* target = pixel(target_row, target_col);
						cv::Vec3b target_color = target->color();
						int total_difference = 0;

						for (int channel = 0; channel < 3; channel++) {
							int difference = abs(target_color[channel] - focus_color[channel]);
							total_difference += difference;
						}

						if (total_difference <= CONNECTION_THRESHOLD) {
							focus->add_connection(target);
						}
					} //end target_in_bounds

				} //end icol
			} //end irow

		} //end col
	} //end row

	timer->end("find connections");
#pragma endregion find connections

#pragma region create regions
	std::cout << "creating regions..." << std::endl;
	timer->begin("create regions");

	for (std::vector<SimplePixel*> row : all_pixels()) {
		for (SimplePixel* pixel : row) {
			if (pixel->parent() == nullptr) {


				SimpleRegion* new_region = new SimpleRegion();
				std::queue<SimplePixel*> pixel_queue;
				pixel_queue.push(pixel);
				pixel->add_to_queue();

				while (!pixel_queue.empty()) {
					SimplePixel* target_pixel = pixel_queue.front();
					pixel_queue.pop();
					target_pixel->set_parent(new_region);
					new_region->add_constituent(target_pixel);

					for (SimplePixel* connection : target_pixel->connections()) {
						if (!connection->in_queue()) {
							pixel_queue.push(connection);
							connection->add_to_queue();

						} //end check if already searched
					} //end search connections
				} //end queue
				add_region(new_region);


			} //end new region
		} //end all pixels 
	} //end all rows

	timer->end("create regions");
#pragma endregion create regions

#pragma region draw regions
	std::cout << "drawing regions..." << std::endl;
	timer->begin("draw regions");
	cv::Mat output(source().size(), source().type());

	for (SimpleRegion* region : regions()) {
		cv::Vec3b avg_color;
		for (int channel = 0; channel < 3; channel++) {
			avg_color[channel] = uchar(region->color_sum()[channel]/region->size());
		}//end find average color

		int height = region->max_row() - region->min_row();
		int width = region->max_col() - region->min_col();
		int row_center = region->min_row() + (height/2);
		int col_center = region->min_col() + (width/2);

		region->set_center(row_center, col_center);
		region->set_color_avg(avg_color);

		cv::Vec3b region_color = colorwheel->get_color();

		for (SimplePixel* pixel : region->constituents()) {
			output.at<cv::Vec3b>(pixel->row(), pixel->col()) = region_color;
		}//end iterate through constituents

	}//end iterate through regions
	set_source(output);

	timer->end("draw regions");
#pragma endregion draw regions

}

void Simplifier::find_edges() {
		cv::Mat output(source().size(), CV_8UC1);
		cv::Canny(source(), output, CANNY_THRESHOLD, CANNY_LOW_THRESHOLD * CANNY_RATIO, CANNY_KERNEL_SIZE);
		set_edges(output);
}

#pragma endregion


#pragma region SimpleRegion

void SimpleRegion::add_constituent(SimplePixel* constituent) {
	int row = constituent->row();
	int col = constituent->col();

	if (row > max_row()) {
		set_max_row(row);
	}
	if (col > max_col()) {
		set_max_col(col);
	}
	if (row < min_row()) {
		set_min_row(row);
	}
	if (col < min_col()) {
		set_min_col(col);
	}
	_constituents.push_back(constituent);
	add_to_color_sum(int(constituent->color()[0]), int(constituent->color()[1]), int(constituent->color()[2])); //conversion to int may be unnecessary
}

#pragma endregion

