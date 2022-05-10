#pragma once
#include"../../includes.h"
//might have to directly include link.h

class Link;
class Clip;
class Moment;
class Frame;
class Antiframe;


class Event {
public:
	Event(int beginning, int end, Clip* parent);
	void load_frames();

#pragma region get-set
	int beginning() { return _beginning; }
	int end() { return _end; }
	int event_id() { return _event_id; }
	Clip* parent() { return _parent; }
	std::vector<Moment*> all_moments() { return _moments; }
	std::vector<Frame*> all_frames() { return _frames; }
	std::vector<Antiframe*> all_antiframes() { return _antiframes; }
	Moment* moment_at(int index) { return _moments[index]; }
	Frame* frame_at(int index) { return _frames[index]; }
	Antiframe* antiframe_at(int index) { return _antiframes[index]; }
	Link* link() { return _link; }
	std::string type() { return _type; }

	void set_type(std::string input) { _type = input; }
	void set_link(Link* input) { _link = input; }
	void set_beginning(int input) { _beginning = input; }
	void set_end(int input) { _beginning = input; }
	void set_event_id(int input) { _event_id = input; }
	void set_parent(Clip* input) { _parent = input; }
	void set_all_moments(std::vector<Moment*> input) { _moments = input; }
	void set_all_frames(std::vector<Frame*> input) { _frames = input; }
	void set_all_antiframes(std::vector<Antiframe*> input) { _antiframes = input; }
	void add_moment(Moment* input) { _moments.push_back(input); }
	void add_frame(Frame* input) { _frames.push_back(input); }
	void add_antiframe(Antiframe* input) { _antiframes.push_back(input); }
#pragma endregion

private:
	std::string _type;
	int _beginning;
	int _end;
	int _event_id;
	Clip* _parent;
	Link* _link;
	std::vector<Moment*> _moments;
	std::vector<Frame*> _frames;
	std::vector<Antiframe*> _antiframes;
};