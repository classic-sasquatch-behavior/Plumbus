#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"

Field::Field(Frame* frame, cv::Mat labels) {
	_frame = frame;
	_labels = labels;
}

Field::~Field() {

}

void Field::calculate_average_region_colors() {
	for (Region* region : all_regions()) {
		std::vector<int> color_sum = { 0,0,0 };

		for (Superpixel* constituent : region->all_constituents()) {
			cv::Vec3b constituent_color = constituent->average_color();
			for (int channel = 0; channel < 3; channel++) {
				color_sum[channel] += constituent_color[channel];
			}
		}

		cv::Vec3b color_out;
		for (int channel = 0; channel < 3; channel++) {
			color_out[channel] = color_sum[channel] / region->num_constituents();
		}
		/*region->set_average_color(color_out);*/
		region->set_average_color(colorwheel->get_color(32));
	}
}

void Field::prune_connections() {

	for (Superpixel* focus : all_superpixels()) {
		std::vector<Superpixel*> new_neighbors;

		cv::Point focus_id = focus->region()->id();
		for (Superpixel* target : focus->all_neighbors()) {
			cv::Point target_id = target->region()->id();

			if (focus_id != target_id && target->region()->num_constituents() > 0) {
				new_neighbors.push_back(target);
			}
		}
		focus->set_neighbors(new_neighbors);
	}
}

#pragma region refine_region_sequence

bool Field::histograms_similar_naive(cv::Mat hist_a, cv::Mat hist_b, int max_threshold, int sum_threshold) { //make it normalize histograms to account for differently sized regions
	bool similar = false;

	cv::Mat hist_a_normalized;
	cv::Mat hist_b_normalized;

	cv::normalize(hist_a, hist_a_normalized, 100.0f, 0.0, cv::NORM_MINMAX);
	cv::normalize(hist_b, hist_b_normalized, 100.0f, 0.0, cv::NORM_MINMAX);

	hist_a = hist_a_normalized;
	hist_b = hist_b_normalized;
	cv::GaussianBlur(hist_a, hist_a, cv::Size(1,3), 0);
	cv::GaussianBlur(hist_b, hist_b, cv::Size(1,3), 0);




	cv::Mat hist_diff(hist_a.size(), hist_a.type());
	cv::absdiff(hist_a, hist_b, hist_diff);
	//std::cout << "hist diff channel values: " << std::endl;
	//for (int i = 0; i < 256; i++) {
	//	std::cout << i << ": " << hist_diff.at<cv::Vec3f>(i)[0] << ", " << hist_diff.at<cv::Vec3f>(i)[1] << ", " << hist_diff.at<cv::Vec3f>(i)[2] << std::endl;
	//}

	std::vector<cv::Mat> channels;
	cv::split(hist_diff, channels);

	cv::Scalar sum_diff_scalar = cv::sum(hist_diff);
	float sum_diff = sum_diff_scalar[0] + sum_diff_scalar[1] + sum_diff_scalar[2];

	double min_diff;
	double max_diff;
	cv::minMaxIdx(hist_diff, &min_diff, &max_diff);

	//std::cout << "max diff: " << max_diff << ", sum diff: " << sum_diff << std::endl;
	//std::cout << std::endl;

	if ((max_diff <= max_threshold)&&(sum_diff <= sum_threshold)) {
		similar = true;
	}

	return similar;
}

//appears to be really slow
bool Field::histograms_similar_BC(cv::Mat hist_a, cv::Mat hist_b, float BC_threshold) { //Bhattacharyya distance
	bool similar = false;

	//prepare histograms
	cv::normalize(hist_a, hist_a, 100.0f, 0.0, cv::NORM_MINMAX);
	cv::normalize(hist_b, hist_b, 100.0f, 0.0, cv::NORM_MINMAX);
	
	cv::GaussianBlur(hist_a, hist_a, cv::Size(1, 3), 0);
	cv::GaussianBlur(hist_b, hist_b, cv::Size(1, 3), 0);


	//calculate Bhattacharyya distance
	float BC_coefficient = 0;

	for (int col = 0; col < 256; col++) {
		for (int channel = 0; channel < 3; channel++) {
			float a_val = hist_a.at<cv::Vec3f>(col)[channel];
			float b_val = hist_b.at<cv::Vec3f>(col)[channel];

			float product = a_val * b_val;
			float root = sqrt(product);
			BC_coefficient += root;
		}
	}

	float BC_distance = std::log((1/BC_coefficient));

	//std::cout << "BC coefficient: " << BC_coefficient << ", BC distance: " << BC_distance << std::endl;

	if (BC_distance < -BC_threshold) {
		similar = true;
	}

	return similar;
}

bool Field::histograms_similar_KL(cv::Mat hist_a, cv::Mat hist_b, float KL_threshold) { //Kullback–Leibler divergence
	bool similar = false;

	//prepare histograms
	cv::normalize(hist_a, hist_a, 100.0f, 0.0, cv::NORM_MINMAX);
	cv::normalize(hist_b, hist_b, 100.0f, 0.0, cv::NORM_MINMAX);

	cv::GaussianBlur(hist_a, hist_a, cv::Size(1, 3), 0);
	cv::GaussianBlur(hist_b, hist_b, cv::Size(1, 3), 0);

	//calculate Kullback–Leibler divergence

	float KL_distance = 0;

	for (int col = 0; col < 256; col++) {
		for (int channel = 0; channel < 3; channel++) {

			float a_val = hist_a.at<cv::Vec3f>(col)[channel];
			float b_val = hist_b.at<cv::Vec3f>(col)[channel];

			float a_over_b = a_val / b_val;
			float log_a_over_b = std::log(a_over_b);
			float a_times_log = a_val * log_a_over_b;
			KL_distance += a_times_log;
		}
	}

	std::cout << "example KL distance: " << KL_distance;








	return similar;
}

void Field::merge_regions(Region* region_keep, Region* region_clear) {

	//absorb = merge and set
	region_keep->absorb_constituents(region_clear->all_constituents()); 
	region_keep->absorb_histogram(region_clear->histogram()); 
	region_keep->absorb_neighbors(region_clear->all_neighboring_regions()); 

	if (region_clear->num_constituents() > 0) { region_clear->clear_constituents(); }
}

void Field::refine_region_sequence() {
	establish_region_neighbors(); 
	init_region_histograms(); 
	calculate_average_region_colors();

	associate_regions_by_histogram(5.45); //5.3 pretty good, but with just a little spillover //5.45 less good but no spillover
	refresh_regions();

	refine_floating_regions();
	refresh_regions();

	//prune_connections();
	//refine_based_on_fitness(10);
	//refresh_regions();

	//refine_floating_regions();
	//refresh_regions();

	//refine_small_regions();


	//associate_regions_by_histogram(5.8);
	//refresh_regions();

	//refine_floating_regions();
	//refresh_regions();

	//associate_regions_by_histogram(5.8);
	//refresh_regions();


	//prune_connections();
	//refine_regions();
	//refresh_regions();
}

void Field::refine_based_on_fitness(int size_threshold) {
	for (Region* region : all_regions()) {
		if (region->num_constituents() > 0 && region->num_constituents() <= size_threshold) {

			std::unordered_map<float, int> region_weights;
			for (Region* neighboring_region : region->all_neighboring_regions()) {
				if (neighboring_region->num_constituents() > 0) {
					region_weights[neighboring_region->id_hash()] = 0;
				}
			}

			for (Superpixel* constituent : region->all_constituents()) {
				for (Superpixel* neighbor : constituent->all_neighbors()) {
					Region* neighboring_region = neighbor->region();

					//find neighbor weights, where weight is how many superpixels are connected to that region
					if (neighboring_region->num_constituents() > 0) {

						float neighboring_region_id = neighboring_region->id_hash();
						region_weights[neighboring_region_id]++;
					}
				}
			}
			
			float most_fitness = -1.0f;
			Region* most_fit_neighbor = nullptr;


			for (Region* neighboring_region : region->all_neighboring_regions()) {
				float neighbor_size = neighboring_region->num_constituents();
				if (neighbor_size > 0) {
					float neighboring_region_id = neighboring_region->id_hash();

					float neighbor_weight = region_weights[neighboring_region_id];


					float neighbor_fitness = neighbor_weight * (100 / neighbor_size);

					if (neighbor_fitness > most_fitness) {
						most_fitness = neighbor_fitness;
						most_fit_neighbor = neighboring_region;
					}
				}
			}

			if (most_fit_neighbor != nullptr) {
				merge_regions(most_fit_neighbor, region);
			}


		}
	}


}


void Field::establish_region_neighbors() {

	for (Region* focus : all_regions()) {
		std::set<Region*> new_neighbors;
		for (Superpixel* constituent : focus->all_constituents()) {
			for (Superpixel* neighbor : constituent->all_neighbors()) {
				new_neighbors.insert(neighbor->region());
			}
		}
		focus->set_neighboring_regions(new_neighbors);
	}
}

void Field::init_region_histograms() { //assumes each region only has one constituent. write a more general function later if you need it, but you probably wont.
	for (Region* region : all_regions()) {
		region->set_histogram(region->constituent_at(0)->histogram());
	}
}

//merges connected regions if the histograms are similar enough. start here.
void Field::associate_regions_by_histogram(float BC_thresh) {

	//gotta figure out where in the cycle to remove dead regions (i.e. regions with num constituents = 0)
	for (Region* focus : all_regions()) {
		if (focus->num_constituents() > 0) {
			for (Region* target : focus->all_neighboring_regions()) {
				if ((target->num_constituents() > 0) && target->id() != focus->id()) {
					cv::Mat focus_hist = focus->histogram();
					cv::Mat target_hist = target->histogram();
					int focus_hist_type = focus_hist.type();
					int target_hist_type = target_hist.type();

					//if (histograms_similar_naive(focus_hist, target_hist, 80, 1000))
					if(histograms_similar_BC(focus_hist, target_hist, BC_thresh))
					{
						merge_regions(focus, target);
					}
				}
			}
		}
	}
}

void Field::refresh_regions() { 
	std::set<Region*> new_regions;

	for (Region* region : all_regions()) {
		if (region->num_constituents() > 0) {
			new_regions.insert(region);
			std::set<Region*> new_neighbors;
			for (Region* neighbor : region->all_neighboring_regions()) {
				if (neighbor->num_constituents() > 0 && (neighbor->id() != region->id())) {
					new_neighbors.insert(neighbor);
				}
			}
			region->set_neighboring_regions(new_neighbors);
		}
	}

	set_regions(new_regions);
}

void Field::refine_floating_regions() {


	for (Region* region : all_regions()) {
		if (region->num_constituents() != 0) {
			if (region->num_neighbors() == 1) {
				Region* neighbor = *region->all_neighboring_regions().begin();
				if (neighbor->num_constituents() > 0) {
					merge_regions(neighbor, region);
				}
			}
		}
	}
}

#pragma endregion

#pragma region refine regions naive

//if size of region is less than threshold, merge with the smallest nearby region. operates purely on size, not on content. maybe I can layer this in as an influencing factor later?
void Field::refine_regions_naive(int threshold) {

	for (Region* region : all_regions()) {
		if (region->num_constituents() <= threshold && region->num_constituents() > 0) {

			cv::Point focus_region_id = region->id();
			Region* smallest_neighbor = nullptr;
			int size_of_smallest_neighbor = INF;

			for (Superpixel* constituent : region->all_constituents()) {
				for (Superpixel* neighbor : constituent->all_neighbors()) {
					cv::Point neighbor_region_id = neighbor->region()->id();


					if (neighbor_region_id != focus_region_id) {
						int size_of_this_neighbor = neighbor->region()->num_constituents();
						if (size_of_this_neighbor < size_of_smallest_neighbor && size_of_this_neighbor != 0) {

							smallest_neighbor = neighbor->region();
							size_of_smallest_neighbor = size_of_this_neighbor;
						}
					}
				}
			}
			merge_regions(smallest_neighbor, region); //smallest neighbor doesnt exist
		}
	}
}

//check if region only has one neighbor. if it does, merge with that neighbor. it didn't seem to work at all, but maybe it's just not working quite right. potentially worth revisiting.
void Field::refine_regions_old() {

	for (Region* region : all_regions()) {
		if (region->num_constituents() != 0) {

			cv::Point region_id = region->id();
			cv::Point solo_id;
			Region* merge_with = nullptr;
			bool found_neighbor = false;
			bool region_invalid = false;

			for (Superpixel* constituent : region->all_constituents()) {

				for (Superpixel* neighbor : constituent->all_neighbors()) {
					cv::Point neighbor_id = neighbor->region()->id();
					if (neighbor_id != solo_id && neighbor_id != region_id) {
						if (!found_neighbor) {
							found_neighbor = true;
							solo_id = neighbor_id;
							merge_with = neighbor->region();
						}
						else {
							region_invalid = true;
							break;
						}
					} //do I want to break here?
				}

				if (region_invalid) {
					break;
				}
			}

			if (!region_invalid) {
				std::cout << "refining regions" << std::endl;
				merge_regions(region, merge_with);
				//merge_with->clear_constituents();
			}
		}
	}
	refresh_regions();
	calculate_average_region_colors();
	for (Region* region : all_regions()) {
		std::cout << "size: " << region->num_constituents() << std::endl;
	}
}

void Field::refine_region_sequence_naive() {
	associate_regions_by_histogram(5);
	calculate_average_region_colors();
	for (int i = 1; i <= 256; i *= 2) {
		prune_connections();
		refine_regions_naive(i);
		refresh_regions();
	}
}

#pragma endregion

void Field::connect_neighbors() {
	std::vector<thrust::pair<int, int>> pairs = GPU->find_borders(labels());

	for (thrust::pair<int, int> pair : pairs) {
		int first_label = pair.first;
		int second_label = pair.second;

		if (first_label != second_label) {
			Superpixel* focus = superpixel_at(first_label);
			Superpixel* target = superpixel_at(second_label);

			focus->add_neighbor(target);
		}
	}
}

void Field::form_regions() {
	int index = 0;
	for (Superpixel* superpixel : all_superpixels()) {
		Region* new_region = new Region(this);
		superpixel->set_region(new_region);
		new_region->add_constituent(superpixel);
		add_region(new_region);
		index++;
	}
}








void Field::affinity_propagation() { //this is gonna take a really, really long time like this lmao
	
	int N = num_superpixels();
	cv::Size matrix_size(N,N);
	cv::Mat similarity_matrix(matrix_size, CV_32SC1);

	//form similarity matrix
	//to speed up: use sparse mats, use gpu gaussian blur
	std::vector<cv::Mat> prepared_hists;
	for (int i = 0; i < N; i++) {
		cv::Mat hist = superpixel_at(i)->histogram();
		cv::normalize(hist, hist, 100.0f);
		cv::GaussianBlur(hist, hist, cv::Size(1, 3), 0);
		prepared_hists.push_back(hist);
	}

	GPU->form_similarity_matrix(prepared_hists, similarity_matrix, N);

	//set diagonal to lowest_val
	cv::Mat similarity_matrix_diagonal = similarity_matrix.diag(0);
	double lowest_val;
	cv::minMaxIdx(similarity_matrix, &lowest_val);
	similarity_matrix_diagonal.setTo((int)lowest_val);

	//2) examine similarity matrix to form responsibility matrix
	cv::Mat responsibility_matrix(matrix_size, CV_32SC1);

	











	//3) examine responsibility matrix to form availibility matrix
	cv::Mat availibility_matrix(matrix_size, CV_32SC1);



	//4) add responsibility matrix and availibility matrix to form critereon matrix 
	cv::Mat critereon_matrix(matrix_size, CV_32SC1);

	//5) examine critereon matrix to identify exemplars


	//6) create regions based on exemplar clusters


}




