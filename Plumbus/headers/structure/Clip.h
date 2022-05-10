#include"../../includes.h"

class Frame;
class Antiframe;
class Moment;
class Shot;
class VirtualEvent;
class VirtualEventLink;
class Link;
class Event;
class Activity;


class Clip {
public:
	Clip(std::string path);
	~Clip();
	void frames_init(); //initialize frames and assign image to each frame object
	void antiframes_init();
	void stitch_moments();
	void moments_init();
	void virtual_events_init();
	void link_virtual_events();
	void evaluate_virtual_events();
	void find_shots();
	void find_objects_within_shots();

	void print_virtual_events();

#pragma region get-set
	std::vector<Frame*> all_frames() { return _frames; }
	std::vector<Antiframe*> all_antiframes() { return _antiframes; }
	std::vector<Moment*> all_moments() { return _moments; }
	Frame* frame_at(int index) { return _frames[index]; }
	Antiframe* antiframe_at(int index) { return _antiframes[index]; }
	Moment* moment_at(int index) { return _moments[index]; }
	std::vector<int> all_difference_values() { return _difference_values; }
	std::vector<Shot*> all_shots() { return _shots; }
	std::vector<VirtualEvent*> all_virtual_events() { return _virtual_events; }
	int difference_value(int index) { return _difference_values[index]; }
	Shot* shot(int index) { return _shots[index]; }
	VirtualEvent* virtual_event(int index) { return _virtual_events[index]; }
	VirtualEvent* virtual_event_head() { return _virtual_event_head; }
	VirtualEventLink* virtual_event_link_head() { return _virtual_event_link_head; }
	std::vector<Event*> all_events() { return _events; }
	Event* event_at(int index) { return _events[index]; }
	Link* event_link_head() { return _event_link_head; }
	std::vector<Activity*> all_activity() { return _activity; }
	Activity* activity_at(int index) { return _activity[index]; }

	void add_activity(Activity* input) { _activity.push_back(input); }
	void add_event(Event* input) { _events.push_back(input); }
	void add_frame(Frame* input) { _frames.push_back(input); }
	void add_antiframe(Antiframe* input) { _antiframes.push_back(input); }
	void add_moment(Moment* input) { _moments.push_back(input); }
	void add_shot(Shot* input) { _shots.push_back(input); }
	void add_virtual_event(VirtualEvent* input) { _virtual_events.push_back(input); }
	void add_difference_value(int input) { _difference_values.push_back(input); }
	void set_virtual_event_head(VirtualEvent* input) { _virtual_event_head = input; }
	void set_virtual_event_link_head(VirtualEventLink* input) { _virtual_event_link_head = input; }
	void set_events(std::vector<Event*> input) { _events = input; }
	void set_event_link_head(Link* input) { _event_link_head = input; }

	int num_moments() { return _moments.size(); }
	int num_frames() { return _frames.size(); }
	int num_antiframes() { return _antiframes.size();  }
	int num_shots() { return _shots.size(); }
	int num_virtual_events() { return _virtual_events.size(); }
	int num_difference_values() { return _difference_values.size(); }
#pragma endregion

private:
	std::vector<Frame*> _frames;
	std::vector<Antiframe*> _antiframes;
	std::vector<Moment*> _moments;
	std::vector<int> _difference_values;
	std::vector<Shot*> _shots;
	std::vector<Activity*> _activity;
	std::vector<Event*> _events;
	std::vector<VirtualEvent*> _virtual_events;
	VirtualEvent* _virtual_event_head;
	VirtualEventLink* _virtual_event_link_head;

	//truly private
	std::string _respath;
	std::queue<Frame*> _frame_queue;
	std::queue<Antiframe*> _antiframe_queue;

	Link* _event_link_head;





};










