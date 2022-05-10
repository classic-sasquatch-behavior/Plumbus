#pragma once
#include"../../includes.h"
#include"Moment.h"
#include"Antiframe.h"

class Moment;
class Antiframe;
class Clip;
class Field;

class Frame : public Moment{
public:
	Frame(cv::Mat source, int frame_index, Clip* parent);
	~Frame();
	void find_antiframes();
	void calculate_layer2();
	void run_filters();
	void identify_local_objects();
	void generate_superpixels(cv::Mat input);
	void find_regions();

#pragma region filters
	//segmentation
	cv::Mat watershed_segmentation(cv::Mat input);
	cv::Mat kmeans_segmentation(cv::Mat input);
	cv::Mat contour_segmentation(cv::Mat input);
	cv::Mat custom_segmentation(cv::Mat input);

	cv::Mat split_segmentation(cv::Mat input); 
	cv::Mat selective_blur_hsv(cv::Mat input);

	cv::Mat draw_superpixels(Field* input);
	cv::Mat draw_regions(Field* input);

	cv::Mat find_edges(cv::Mat input);
	cv::Mat binary_overlay(cv::Mat base_in, cv::Mat mask_in );
#pragma endregion

#pragma region get-set
	//get images
	inline cv::Mat edge_overlay() { return _edge_overlay; }
	inline cv::Mat super_edges() { return _super_edges; }
	inline cv::Mat edges() { return _edges; }
	inline cv::Mat superpixels() { return _superpixels; }
	inline cv::Mat source() { return _source; }
	cv::Mat past_difference();
	cv::Mat future_difference();
	inline cv::Mat blurred() { return _blurred; }
	inline cv::Mat regions() { return _regions; }

	//get related objects
	inline Antiframe* past() { return _past; }
	inline Antiframe* future() { return _future; }
	inline Clip* parent() { return _parent; }
	inline Field* field() { return _field; }

	//get information
	inline int frame_index() { return _frame_index; }

	//set related objects
	void set_past(Moment* input) override; 
	void set_future(Moment* input) override;
	inline void set_field(Field* input) { _field = input; }

#pragma endregion



private:
	//information
	int _frame_index;

	//related objects
	Antiframe* _past = nullptr;
	Antiframe* _future = nullptr;
	Clip* _parent;
	Field* _field;

	//images
	cv::Mat _source;
	cv::Mat _blurred;
	cv::Mat _superpixels;
	cv::Mat _edges;
	cv::Mat _super_edges;
	cv::Mat _edge_overlay;
	cv::Mat _regions;

};