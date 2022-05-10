#include "../../includes.h"
#include "../../classes.h"
#include"../../config.h"

#pragma region structors

Shot::Shot(int beginning, int end, Clip* parent) : Event(beginning, end, parent) {

}

Shot::~Shot() {

}

#pragma endregion

void Shot::find_objects_within_frames() {
	for (Frame* frame : all_frames()) {
		frame->identify_local_objects();
	}
}

void Shot::link_objects_between_frames() {

}