#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"

#pragma region structors

Clip::Clip(std::string path) {
	_respath = path;

								//timer->begin("frames init (init clip)");
	frames_init();				//frames init
								//timer->end("frames init (init clip)");

								//timer->begin("antiframes init (init clip)");
	antiframes_init();			//antiframes init
								//timer->end("antiframes init (init clip)");

								//timer->begin("stitching moments (init clip)");
	stitch_moments();			//stitch moments
								//timer->end("stitching moments (init clip)");

								//timer->begin("moments init (init clip)");
	moments_init();				//moments init
								//timer->end("moments init (init clip)");

								//timer->begin("virtual events init (init clip)");
	//virtual_events_init();	//events init
								//timer->end("virtual events init (init clip)");

								//timer->begin("link virtual events (init clip)");
	//link_virtual_events();		//link events
								//timer->end("link virtual events (init clip)");

								//timer->begin("evaluate virtual events (init clip)");
	//evaluate_virtual_events();	//evaluate events
								//timer->end("evaluate virtual events (init clip)");

								//timer->begin("find shots (init clip)");
	//find_shots();				//find shots
								//timer->end("find shots (init clip)");

								//timer->begin("find objects within shots (init clip)");
	//find_objects_within_shots();//find objects within shots
								//timer->end("find objects within shots (init clip)");
						
								//timer->begin("print virtual events (init clip)");
	//print_virtual_events();		//print events
								//timer->end("print virtual events (init clip)");
}

Clip::~Clip() {

}

#pragma endregion

void Clip::frames_init() {
	int frame_index = 0;
	for (const auto& file : fs::directory_iterator(_respath)) {
		std::cout << "initializing frame " << frame_index << std::endl;
		std::string file_path = file.path().string();
		cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);

		Frame* new_frame = new Frame(image, frame_index, this); //frame constructor
		_frames.push_back(new_frame);
		_frame_queue.push(new_frame);
		frame_index++;
	}
}

void Clip::antiframes_init() { //create antiframe objects
	for (int antiframe_index = 0; antiframe_index < num_frames() - 1; antiframe_index++) {
		std::cout << "initializing antiframe " << antiframe_index << std::endl;
		Antiframe* new_antiframe = new Antiframe(antiframe_index, this);
		_antiframes.push_back(new_antiframe);
		_antiframe_queue.push(new_antiframe);
	}
}

void Clip::stitch_moments() {
	int flip = 0;
	while (!_frame_queue.empty() || !_antiframe_queue.empty()) {
		switch (flip) {
		case 0: _moments.push_back(_frame_queue.front()); _frame_queue.pop(); flip++; break;
		case 1: _moments.push_back(_antiframe_queue.front()); _antiframe_queue.pop(); flip--; break;
		default:break;
		}
	}

	for (int moment_index = 0; moment_index < num_moments(); moment_index++) { // I think this will have to be < num_moments() +- 1
		Moment* this_moment = moment_at(moment_index);
		this_moment->set_moment_id(moment_index);
		if (moment_index > 0) {
			Moment* past_moment = moment_at(moment_index - 1);
			this_moment->set_past(past_moment);
		}
		if (moment_index < num_moments() - 1) {
			Moment* future_moment = moment_at(moment_index + 1);
			this_moment->set_future(future_moment);
		}
	}
}

void Clip::moments_init() {
	//timer->begin("running antiframes (moments init)");
	//for (Antiframe* antiframe : all_antiframes()) {
	//	std::cout << "running antiframe " << antiframe->antiframe_index() << std::endl;
	//	antiframe->run_filters();
	//}
	//timer->end("running antiframes (moments init)");

	//timer->begin("running frames (moments init)");
	for (Frame* frame : all_frames()) {
		std::cout << "running frame " << frame->frame_index() << std::endl;
		timer->begin("run frame");
		frame->run_filters();
		timer->end("run frame");
	}
	//timer->end("running frames (moments init)");
}

#pragma region virtual_events_init util

VirtualEvent* begin_event(std::vector<int> &current_sequence, Clip* parent, std::string type, int antiframe, int &event_id) {
	int beginning_frame = 0;
	if (current_sequence.size() <= 1) {
		beginning_frame = antiframe;
	}
	else {
		beginning_frame = antiframe + 1;
	}
	VirtualEvent* current_event = new VirtualEvent(parent, type);
	current_event->set_beginning(beginning_frame);
	parent->add_virtual_event(current_event);
	current_event->set_event_id(event_id);
	event_id++;

	return current_event;
}

void end_event(VirtualEvent* current_event, std::vector<int> &current_sequence, int antiframe) {
	int end_frame = 0;
	if (current_sequence.size() == 1) {
		end_frame = antiframe;
	}
	else {
		end_frame = antiframe - 1;
	}
	current_event->set_differences(current_sequence);
	current_event->set_end(end_frame);
	current_sequence.clear();
}

#pragma endregion
void Clip::virtual_events_init() {

	VirtualEvent* current_shot_event = nullptr;
	std::vector<int> current_shot_sequence; 

	VirtualEvent* current_activity_event = nullptr;
	std::vector<int> current_activity_sequence; 

	int event_id = 0;
	bool active;

	if (difference_value(0) == 0) {
		current_shot_event = begin_event(current_shot_sequence, this, "shot", 0, event_id);
		active = false;
	} else {
		current_activity_event = begin_event(current_activity_sequence, this, "activity", 0, event_id);
		active = true;
	}

	for (int antiframe = 0; antiframe < all_difference_values().size(); antiframe++) {
		int present_value = int(difference_value(antiframe));

		switch (present_value) {
		case 0:
			current_shot_sequence.push_back(present_value);
			if (active) { //falling edge
				active = false;
				current_shot_event = begin_event(current_shot_sequence, this, "shot", antiframe, event_id);
				end_event(current_activity_event, current_activity_sequence, antiframe);
			} break;

		default:
			current_activity_sequence.push_back(present_value);
			if (!active) { //rising edge
				active = true;
				end_event(current_shot_event, current_shot_sequence, antiframe);
				current_activity_event = begin_event(current_activity_sequence, this, "activity", antiframe, event_id);
			} break;
		}
	}

	if (active) {
		end_event(current_activity_event, current_activity_sequence, all_difference_values().size() - 1);
	} else {
		end_event(current_shot_event, current_shot_sequence, all_difference_values().size() - 1);
	}


}

void Clip::link_virtual_events() {
	for (VirtualEvent* this_event : all_virtual_events()) {

		VirtualEventLink* this_event_link = this_event->link();
		VirtualEventLink* previous_event_link = nullptr;
		VirtualEventLink* next_event_link = nullptr;
		if (this_event->event_id() != 0) {
			previous_event_link = virtual_event(this_event->event_id() - 1)->link();
		}
		if (this_event->event_id() != num_virtual_events() - 1 ) {
			next_event_link = virtual_event(this_event->event_id() + 1)->link();
		}
		this_event->link()->set_previous_link(previous_event_link);
		this_event->link()->set_next_link(next_event_link);
	}
	set_virtual_event_head(virtual_event(0));
	set_virtual_event_link_head(virtual_event_head()->link()); //check what's actually in 1131 to 1134 (event 53)
}

void Clip::evaluate_virtual_events() {
	VirtualEventLink* target = virtual_event_link_head();

	while (target != nullptr) {
		if (target->owner()->beginning() == 0) {
			set_virtual_event_link_head(target);
		}
		target = target->owner()->self_evaluate();
	}


}

void Clip::find_shots() {
	VirtualEventLink* target = virtual_event_link_head();
	int event_id = 0;
	Link* new_event_link_head = nullptr;
	while (target != nullptr) {
		VirtualEvent* target_event = target->owner();
		Link* previous_link = nullptr;


		if (target_event->type() == "shot") { //create shot
			Shot* new_shot = new Shot(target_event->beginning(), target_event->end(), this);
			new_shot->set_event_id(event_id);
			event_id++;
			new_shot->set_type("shot");
			#pragma region set hard cuts
			if (target_event->previous_event() != nullptr){
				if (target_event->previous_event()->type() == "hard cut") {
					new_shot->set_hard_cut_before(true);
				} else {
					new_shot->set_hard_cut_before(false);
				}
			}

			if (target_event->next_event() != nullptr) {
				if (target_event->next_event()->type() == "hard cut") {
					new_shot->set_hard_cut_after(true);
				}
				else {
					new_shot->set_hard_cut_after(false);
				}
			}
			#pragma endregion

			//set previous link and tie last link
			new_shot->link()->set_previous_link(previous_link);
			if (previous_link != nullptr) { previous_link->set_next_link(new_shot->link()); }
			previous_link = new_shot->link();
			if (new_event_link_head == nullptr) { new_event_link_head = new_shot->link(); }
			add_event(new_shot);
			add_shot(new_shot);
		} //end create shot


		else if (target_event->type() == "activity") { //create activity
			Activity* new_activity = new Activity(target_event->beginning(), target_event->end(), this);
			new_activity->set_event_id(event_id);
			event_id++;
			new_activity->set_type("activity");
			//set previous link and tie last link
			new_activity->link()->set_previous_link(previous_link);
			if (previous_link != nullptr) { previous_link->set_next_link(new_activity->link()); }
			previous_link = new_activity->link();
			if (new_event_link_head == nullptr) { new_event_link_head = new_activity->link(); }

			add_event(new_activity);
			add_activity(new_activity);
		}//end create activity

		else if (target_event->type() == "hard cut") {} //skip hard cut

		else { //type error
			std::cout << std::endl;
			std::cout << "ERROR: " << target_event->type() << " is not a recognized type (event at " << target_event->beginning() << " - " << target_event->end() << ")." << std::endl;
			std::cout << std::endl;
		} //end type error


		target = target->next_link();
	}

	set_event_link_head(new_event_link_head);

}

void Clip::find_objects_within_shots() {

	for (Shot* shot : all_shots()) {
		shot->find_objects_within_frames();
		shot->link_objects_between_frames();
	}
	
	

}

void Clip::print_virtual_events() {
	std::ofstream event_log;
	event_log.open("event_log.txt");
	event_log << "";
	event_log.close();

	VirtualEventLink* target = virtual_event_link_head();
	while (target != nullptr) {
		event_log.open("event_log.txt", std::fstream::app);
		event_log << "Event " << target->owner()->event_id() << ": length - " << target->owner()->size() << ", from " << target->owner()->beginning() << " to " << target->owner()->end() << ", " << target->owner()->type();

		event_log << ", sequence: ";
		for (int i = 0; i < target->owner()->size(); i++) {
			event_log << target->owner()->difference_at(i);
			if (i != target->owner()->size() - 1) {
				event_log << ", ";
			}
		}
		
		event_log << std::endl;
		target = target->next_link();
		event_log.close();
	}

}





