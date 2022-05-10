#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"



Event::Event(int beginning, int end, Clip* parent) {
	_beginning = beginning;
	_end = end;
	_parent = parent;
	_link = new Link(this);
	load_frames();
}

void Event::load_frames() {
	
	//load all frames
	for (int frame = beginning(); frame <= end(); frame++) {
		add_frame(parent()->frame_at(frame));
	}

	//load all antiframes
	for (int antiframe = beginning(); antiframe <= end() - 1; antiframe++) {
		add_antiframe(parent()->antiframe_at(antiframe));
	}

	//load all moments
	for (int moment = beginning() * 2; moment <= end() * 2; moment++) {
		int moment_index = moment - (beginning()*2);
		add_moment(parent()->moment_at(moment));
		moment_at(moment_index)->set_event_parent(this);
	}
}