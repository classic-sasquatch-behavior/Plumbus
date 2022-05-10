#pragma once
#include"../../includes.h"

class Moment;
class Clip;
class Frame;
class Antiframe;
class VirtualEventLink;


class VirtualEvent {
public:
	VirtualEvent(Clip* parent, std::string type);
	~VirtualEvent();
	VirtualEventLink* self_evaluate();
	VirtualEventLink* split(std::vector<int> places, std::vector<std::string> split_types);
	VirtualEventLink* merge(bool merge_previous, bool merge_next, std::string merge_type);

#pragma region get-set
	//shot, activity, light noise, dark noise, hard cut
	VirtualEventLink* link() { return _link; }
	std::string type() { return _type; }
	int beginning() { return _beginning; }
	int end() { return _end; }
	Clip* parent() { return _parent; }
	std::vector<Moment*> all_moments() { return _moments; }
	std::vector<Frame*> all_frames() { return _frames; }
	std::vector<Antiframe*> all_antiframes() { return _antiframes; }
	Moment* moment(int index) { return _moments[index]; }
	Frame* frame(int index) { return _frames[index]; }
	Antiframe* antiframe(int index) { return _antiframes[index]; }
	std::vector<int> all_differences() { return _differences; }
	int event_id() { return _event_id; }
	int difference_at(int index) { return _differences[index]; }
	VirtualEvent* previous_event();
	VirtualEvent* next_event();
	bool split_queued() { return _split_queued; }
	std::vector<int> split_locations() { return _split_locations; }
	std::vector<std::string> split_types() { return _split_types; }

	void set_link(VirtualEventLink* input) { _link = input; }
	void queue_split() { _split_queued = true; }
	void add_split_location(int input) { _split_locations.push_back(input); }
	void add_split_type(std::string input) { _split_types.push_back(input); }

	void set_beginning(int input) { _beginning = input; }
	void set_end(int input) { _end = input; }
	void set_parent(Clip* input) { _parent = input; }
	void set_type(std::string input) { _type = input; }
	void set_differences(std::vector<int> input) { _differences = input; }
	void set_moments(std::vector<Moment*> input) { _moments = input; }
	void set_frames(std::vector<Frame*> input) { _frames = input; }
	void set_antiframes(std::vector<Antiframe*> input) { _antiframes = input; }
	void add_moment(Moment* input) { _moments.push_back(input); }
	void add_frame(Frame* input) { _frames.push_back(input); }
	void add_antiframe(Antiframe* input) { _antiframes.push_back(input); }
	void set_event_id(int input) { _event_id = input; }

	int size() { return _differences.size(); }
	int num_moments() { return _moments.size(); }
	int num_frames() { return _frames.size(); }
	int num_antiframes() { return _antiframes.size(); }
#pragma endregion

private:
	std::string _type = "undefined"; 
	int _beginning;
	int _end;
	Clip* _parent;
	int _event_id;

	bool _split_queued = false;
	bool _merge_queued = false;

	std::vector<int> _split_locations;
	std::vector<std::string> _split_types;

	VirtualEventLink* _link;

	std::vector<int> _differences;
	std::vector<Moment*> _moments;
	std::vector<Frame*> _frames;
	std::vector<Antiframe*> _antiframes;
};

class VirtualEventLink {
public:
	VirtualEventLink(VirtualEvent* owner);

#pragma region get-set

	VirtualEvent* owner() { return _owner;  }
	VirtualEventLink* previous_link() { return _previous_link; }
	VirtualEventLink* next_link() { return _next_link; }

	void set_owner(VirtualEvent* input) { _owner = input; }
	void set_previous_link(VirtualEventLink* input) { _previous_link = input; }
	void set_next_link(VirtualEventLink* input) { _next_link = input; }

#pragma endregion


private:
	VirtualEvent* _owner = nullptr;
	VirtualEventLink* _previous_link = nullptr;
	VirtualEventLink* _next_link = nullptr;


};