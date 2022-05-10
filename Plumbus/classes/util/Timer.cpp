#include"../../includes.h"
#include"../../classes.h"
#include"../../config.h"




//-----timer-----//

#pragma region Timer



Timer::Timer() {

}

void Timer::check(std::string origin) {
	std::cout << std::endl;
	std::cout << "timer accessed from " << origin << ", pointer: " << this << std::endl;
	std::cout << std::endl;
}

void Timer::begin(std::string name) {
	if (_timed_events.find(name) == _timed_events.end()) {
		TimedEvent* new_event = new TimedEvent(name);
		_timed_events[name] = new_event;
	}
	_timed_events.at(name)->begin_event();
}

void Timer::end(std::string name) {
	_timed_events.at(name)->end_event();
}

void Timer::generate_report() {
	std::cout << std::endl;
	for (auto & [name, event] : _timed_events) {
		event->generate_stats();

		if (event->average_time() == -1) {
			std::cout << "ERROR: event " + name + " never ran" << std::endl;
		} else {
			int times_run = event->times_run();
			float average_time = event->average_time();
			float longest_time = event->longest_time();
			float overall_time = average_time * times_run;
			float average_seconds = average_time / 1000.0f;
			float longest_seconds = longest_time / 1000.0f;
			float overall_seconds = overall_time / 1000.0f;

			std::cout << "Event " + name + ": " << std::endl;
			std::cout << name + " ran " << times_run << " times" << std::endl;
			std::cout << "average runtime of " + name + ": " << average_time << " ms (" << average_seconds << " seconds)" << std::endl;
			std::cout << "longest runtime of " + name + ": " << longest_time << " ms (" << longest_seconds << " seconds)" << std::endl;
			std::cout << "overall time used by " + name + ": " << overall_time << " ms (" << overall_seconds << " seconds)" << std::endl;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}



#pragma endregion

//-----events-----//

#pragma region TimedEvent

TimedEvent::TimedEvent(std::string name) {
	_name = name;
}

void TimedEvent::begin_event() {
	_times_run++;
	_start = std::chrono::system_clock::now();
}

void TimedEvent::end_event() {
	float time_elapsed = 0;
	std::chrono::duration<double, std::milli> duration;
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	duration = end - _start;
	time_elapsed = duration.count();

	if (time_elapsed > _longest_time) {
		_longest_time = time_elapsed;
	}
	_times.push_back(time_elapsed);
}

void TimedEvent::generate_stats() {
	float sum_times = 0;
	for (float time : _times) {
		sum_times += time;
	}

	if (_times.size() == 0) {
		_average_time = -1;
	} else {
		_average_time = sum_times / _times.size();
	}
}


#pragma endregion




