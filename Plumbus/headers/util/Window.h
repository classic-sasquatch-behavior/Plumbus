#include"../../includes.h"



class Clip;
class Frame;


class Window {
public:
	Window(int WINDOW_HEIGHT, int WINDOW_WIDTH, Clip* clip);
	void change_frame(int direction);
	void update_window(std::string name, cv::Mat image);
	Frame* frame() { return _frame; }
	void increment_plane(int direction);
	int plane_index() { return _planeIndex; }
	cv::Mat overlay(cv::Mat input_a, cv::Mat input_b);

private:
	int _width;
	int _height;
	int _frameIndex = 0;
	int _planeIndex = 0;
	Frame* _frame;
	Clip* _clip;
};