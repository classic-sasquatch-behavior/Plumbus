#include"../../includes.h"

class Timer;
class TimedEvent;

class Timer {
public:
	Timer();
	void begin(std::string name);
	void end(std::string name);
	TimedEvent* get_event(std::string name) { return _timed_events.at(name); }
	void check(std::string origin);

	void generate_report();

private:
	std::unordered_map<std::string, TimedEvent*> _timed_events;
};




class TimedEvent {
public:
	TimedEvent(std::string name);
	void begin_event();
	void end_event();
	void generate_stats();

	int times_run() { return _times_run; }
	float average_time() { return _average_time; }
	float longest_time() { return _longest_time; }

private:
	std::chrono::time_point<std::chrono::system_clock> _start;
	int _times_run = 0;
	float _average_time;
	float _longest_time = 0;
	std::vector<float> _times;
	std::string _name;
};


