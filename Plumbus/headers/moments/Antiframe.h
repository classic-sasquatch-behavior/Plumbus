#pragma once
#include"../../includes.h"
#include"Moment.h"
#include"Frame.h"

class Moment;
class Frame;
class Clip;

class Antiframe : public Moment{
public:
	Antiframe(int antiframe_index, Clip* parent);
	void run_filters();
	void calculate_layer1();
	void quantify_difference();

#pragma region get-set
	Clip* parent() { return _parent; }
	cv::Mat layer1() { return _layer1; }
	Frame* past() { return _past; }
	Frame* future() { return _future; }
	int antiframe_index() { return _antiframe_index; }
	int difference_ratio() { return _difference_ratio; }

	void set_past(Moment* input) override; 
	void set_future(Moment* input) override; 

#pragma endregion



private:
	int _antiframe_index;
	cv::Mat _layer1;
	Frame* _past = nullptr;
	Frame* _future = nullptr;
	Clip* _parent;

	float _difference_ratio; //white pixels per black pixel (#white/#black)
};


