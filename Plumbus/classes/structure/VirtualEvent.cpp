#pragma once
#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"




#pragma region structors

VirtualEvent::VirtualEvent(Clip* parent, std::string type) {
	_parent = parent;
	_type = type;
	_link = new VirtualEventLink(this);
}

VirtualEvent::~VirtualEvent() {
	delete link();
}

VirtualEventLink::VirtualEventLink(VirtualEvent* owner) {
	_owner = owner;
}



#pragma endregion

#pragma region get-set
VirtualEvent* VirtualEvent::next_event() {
	if (_link->next_link() == nullptr) { return nullptr; }
	else { return _link->next_link()->owner(); }
}

VirtualEvent* VirtualEvent::previous_event() {
	if (_link->previous_link() == nullptr) {return nullptr;}
	else { return _link->previous_link()->owner(); }
}
#pragma endregion


VirtualEventLink* VirtualEvent::self_evaluate() {
	VirtualEventLink* check_next = link()->next_link();
	if (size() == 1) { //size == 1 and unprocessed
		if (difference_at(0) == 1 ) {
			set_type("light noise");
		} else if (difference_at(0) == 0) {
			set_type("dark noise");
		} else {
			set_type("hard cut");
		}

		//previous_event()->set_end(previous_event()->end() + 1);
		previous_event()->set_end(beginning());//if it breaks, it would probably be breaking because we're affecting size 1 events, or something to do with the beginning value

	}
	else if (previous_event() != nullptr && ((type() == "shot" || type() == "activity") && (previous_event()->type() == "shot" || previous_event()->type() == "activity"))) {
		previous_event()->set_end(beginning() - 1);
	}

	if(type() == "activity"){
		std::vector<int> split_at;
		std::vector<std::string> split_types;
		int max_val = 0;
		for (int i = 0; i < size(); i++) { //iterate through values
		int previous_value = 0;
		if (link()->previous_link() != nullptr) {
			previous_value = previous_event()->difference_at(previous_event()->size() - 1);
		}
		int this_value = difference_at(i);
		int next_value = 0;
		if (link()->next_link() != nullptr) {
			next_value = next_event()->difference_at(0);
		}

		if (this_value > max_val) {
			max_val = this_value;
		}

		#pragma region border check

		if (i != 0) { 
			previous_value = difference_at(i - 1);
		} 
		if (i != size() - 1) {
			next_value = difference_at(i + 1);
		}

		#pragma endregion load previous and next values

			if ((this_value > (previous_value + 1) * 3)&&(this_value > (next_value + 1) * 3)) {
				split_at.push_back(i);
				split_types.push_back("hard cut");
			}
		} //end iterate through values

		if (max_val <= 1) {
			set_type("weak activity");
		}
		if (split_at.size() > 0) {
			return split(split_at, split_types);
		}
	} //end if activity

#pragma region typechecks
	std::string previous_type = "undefined";
	std::string next_type = "undefined";
	if (previous_event() != nullptr) {
		previous_type = previous_event()->type();
	}
	if (next_event() != nullptr) {
		next_type = next_event()->type();
	}

	if (type() == "light noise" || type() == "dark noise" || type() == "mixed noise") {


		if (previous_type == next_type) { 
			return merge(true, true, previous_type);
		}
		else if (previous_type == "light noise" || previous_type == "dark noise" || previous_type == "mixed noise") {
			return merge(true, false, "mixed noise");
		}
		else if (next_type == "light noise" || next_type == "dark noise" || next_type == "mixed noise") {
			return merge(false, true, "mixed noise");
		}
		else if (previous_type == "activity") {
			return merge(true, false, "activity");
		}
		else if (next_type == "activity") {
			return merge(false, true, "activity");
		}
		else if (previous_type == "shot") {
			return merge(true, false, "shot");
		}
		else if (next_type == "shot") {
			return merge(false, true, "shot");
		}
	}

	if (type() == "weak activity") { //if it's not obvious what the problem is, comment out from here down when debugging
		if (previous_type == "shot" && next_type == "shot") {
			return merge(true, true, "shot");
		}
		if (previous_type == "shot") {
			return merge(true, false, "shot");
		}
		if (next_type == "shot") {
			return merge(false, true, "shot");
		}
	}

	if (type() == "activity") {
		if (previous_type == "hard cut" && next_type == "hard cut") {
			set_type("shot");
		}
	}



#pragma endregion



	return check_next;
}



VirtualEventLink* VirtualEvent::split(std::vector<int> places, std::vector<std::string> split_types) {
	VirtualEventLink* return_link = nullptr;
	VirtualEventLink* old_previous_link = link()->previous_link();

	VirtualEventLink* original_next_link = link()->next_link();


	int relative_beginning = 0;
	int relative_end = -1;
	std::vector<int>new_values;
	int split_at = 0;

	for (int split_it = 0; split_it < places.size(); split_it++) {
		split_at = places[split_it];
		std::string split_type = split_types[split_it];
		relative_beginning = relative_end + 1;
		relative_end = split_at;
		if (split_at != 0) {
			VirtualEvent* new_event = new VirtualEvent(parent(), "activity");

			//set beginning and end //done

			new_event->set_beginning(this->beginning() + relative_beginning);
			new_event->set_end(this->beginning() + relative_end);

			//set up links //done
			VirtualEventLink* new_this_link = new_event->link();
			new_this_link->set_previous_link(old_previous_link);
			old_previous_link->set_next_link(new_this_link);
			old_previous_link = new_this_link;
			if (split_it == 0) {
				return_link = new_this_link;
			}

			//set content

			int event_length = relative_end - relative_beginning;
			for (int i = 0; i < event_length; i++) {
				new_values.push_back(difference_at(relative_beginning + i));
			}
			new_event->set_differences(new_values);
			new_values.clear();
		} 
		
		if (split_type == "hard cut") {

			VirtualEvent* new_cut = new VirtualEvent(parent(), "hard cut");

			//set beginning and end //done
			new_cut->set_beginning(this->beginning() + split_at);
			new_cut->set_end(this->beginning() + split_at + 1);

			//set up links //done
			VirtualEventLink* new_cut_link = new_cut->link();
			new_cut_link->set_previous_link(old_previous_link);
			old_previous_link->set_next_link(new_cut_link);
			old_previous_link = new_cut_link;

			//set content
			new_values.push_back(difference_at(split_at));
			new_cut->set_differences(new_values);
			new_values.clear();

			if (split_at == 0) {
				return_link = new_cut_link;
			}
			if (split_at == size() - 1) {
				new_cut_link->set_next_link(original_next_link);
			}
		}

	}

	if (split_at != size() - 1) {
		//once loop is done, create final event and tie it to the original next event
		VirtualEvent* final_new_event = new VirtualEvent(parent(), "activity");


		//set beginning and end //done
		relative_beginning = relative_end + 1;
		final_new_event->set_beginning(this->beginning() + relative_beginning);
		final_new_event->set_end(this->end());

		//set up links //done
		VirtualEventLink* final_link = final_new_event->link();
		old_previous_link->set_next_link(final_link);
		final_link->set_previous_link(old_previous_link);
		final_link->set_next_link(original_next_link);
		original_next_link->set_previous_link(final_link);


		//set content
		for (int i = relative_beginning; i < size(); i++) {
			new_values.push_back(difference_at(i));
		}
		final_new_event->set_differences(new_values);
		new_values.clear();
	}


	return return_link;
}

VirtualEventLink* VirtualEvent::merge(bool merge_previous, bool merge_next, std::string merge_type) {
	VirtualEvent* previous = previous_event();
	VirtualEvent* next = next_event();
	VirtualEvent* previous_previous = previous->previous_event();
	VirtualEvent* next_next = next->next_event();
	
	VirtualEvent* new_event = new VirtualEvent(parent(), merge_type);
	VirtualEventLink* new_event_link = new_event->link();
	//set beginning, end, difference values, link previous and next events, link previous and next events, link new event to previous previous and next next
	int new_beginning = beginning();
	int new_end = end();
	std::vector<int> new_differences;
	VirtualEvent* new_previous = previous;
	VirtualEvent* new_next = next;

	std::vector<int> previous_differences = previous->all_differences();
	std::vector<int> this_differences = all_differences();
	std::vector<int> next_differences = next->all_differences();

	new_differences.reserve(previous_differences.size() + this_differences.size() + next_differences.size());

	if (merge_previous) {

		//new fix
		//if (size() == 1) { //might want to check if mixed noise as well
		//	new_end--;
		//}
		//new fix

		new_beginning = previous->beginning();
		new_previous = previous_previous;
		new_differences.insert(new_differences.end(), previous_differences.begin(), previous_differences.end());
		delete previous;
	}

	new_differences.insert(new_differences.end(), this_differences.begin(), this_differences.end());

	if (merge_next) { 
		new_end = next->end();

		//new fix
		//if (next->size() == 1) { //might want to check if mixed noise as well
		//	new_end--;
		//}
		//new fix

		new_next = next_next;
		new_differences.insert(new_differences.end(), next_differences.begin(), next_differences.end());
		delete next;
	}

	new_event->set_beginning(new_beginning); //breakpoint here
	new_event->set_end(new_end);
	new_event->set_differences(new_differences);



	if (new_next != NULL) {
		new_event->link()->set_next_link(new_next->link());
		new_next->link()->set_previous_link(new_event_link);
	}
	else {
		new_event->link()->set_next_link(nullptr);
	}

	if (new_previous != NULL) {
		new_event->link()->set_previous_link(new_previous->link());
		new_previous->link()->set_next_link(new_event_link);
	}
	else {
		new_event->link()->set_previous_link(nullptr);
	}



	//delete this		//after program runs okay, try this
	return new_event_link;
}