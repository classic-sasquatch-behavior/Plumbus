#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"


class Event;


Window::Window(int WINDOW_HEIGHT, int WINDOW_WIDTH, Clip* clip) {
	_width = WINDOW_WIDTH;
	_height = WINDOW_HEIGHT;
	_clip = clip;
	_frame = _clip->frame_at(_frameIndex);


	
}



void Window::increment_plane(int direction) {
	_planeIndex = (_planeIndex + 3 + direction) % 3;
	std::cout << "plane " << _planeIndex << " of 3" << std::endl;
}

cv::Mat Window::overlay(cv::Mat input_a, cv::Mat input_b) {
	cv::Mat result;
	if (input_a.type() == CV_8UC1) {
		cv::cvtColor(input_a, input_a, cv::COLOR_GRAY2BGRA);
	}
	if (input_b.type() == CV_8UC1) {
		cv::cvtColor(input_b, input_b, cv::COLOR_GRAY2BGRA);
	}
	if (input_a.type() == CV_8UC3) {
		cv::cvtColor(input_a, input_a, cv::COLOR_BGR2BGRA);
	}
	if (input_b.type() == CV_8UC3) {
		cv::cvtColor(input_b, input_b, cv::COLOR_BGR2BGRA);
	}
	double alpha = 0.5; //alpha goes from 0 to 1
	double beta = (1.0 - alpha);
	addWeighted(input_a, alpha, input_b, beta, 0.0, result);

	return result;
}

void Window::change_frame(int direction) {
	_frameIndex = (_frameIndex + _clip->num_frames() + direction) % _clip->num_frames();
	_frame = _clip->frame_at(_frameIndex);
	
	//Event* event_parent = _clip->frame_at(_frameIndex)->event_parent();
	//std::cout << "frame: " << _frameIndex << " ratio: " << _clip->frame_at(_frameIndex)->future()->difference_ratio() << " shot id: " << event_parent->event_id() << " shot type: " << event_parent->type() << std::endl;
	std::cout << "frame: " << _frameIndex << std::endl;
}

void Window::update_window(std::string name, cv::Mat image) {
	cv::imshow(name, image);
}




