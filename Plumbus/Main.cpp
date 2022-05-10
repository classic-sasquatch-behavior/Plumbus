#include"includes.h"
#include"classes.h"
#include"config.h"


#pragma region documentation

//I am going to document my code like this. basically just what I was saying in the
//google doc. in fact, let me just copypasta it over.
//
//Plumbus is a collection of transformers and neural networks which extract 
//structural - temporal data from episodes of the tv show "How It's Made" 
//in order to produce novel pieces of video media which simulate unique
//episodes of the show, based on nonsensical objects but with the style
//and flow of the How It's Made series. The information from the dataset 
//will first be cast in conceptual space, from which several models for
//large scale features will be trained (i.e. scene length, number of scenes,
//and so on). These models will inform the convolutional neural networks
//to reproduce the fine-grain visual characteristics of the data. Between 
//these two will be the models which garuntee temporal coherence.



//The project can thus be split into four stages :
//		Processing the dataset
//		Designing the structural models
//		Designing the time-coherence models
//		Training the visual model
// 
//Maintaining the visual coherence of objects through time is the most 
//difficult challenge facing time - dependent generative media, and 
//will be of great importance here as well.Navigating this project 
//successfully will come down to maintaining coherence on all timescales, 
//from individual frames to the overall plot of the video, and accurately 
//reproducing the influence of these timescales upon one another. The models
//will have to cascade into one another, while at the same time providing feedback
//which affects any lower levels of the model. Therefore, care must be taken
//to make all the models more or less interoperable.
//
//At present, I am still working on processing the dataset. I hope to come away
//with the following information quantified:
//		shots
//		scenes
//		objects within frame
//		objects within shot
//		objects within scene

#pragma endregion 1) project overview

int WINDOW_WIDTH = 1600;
int WINDOW_HEIGHT = 900;
std::string RES_PATH = "C:/Users/Thelonious/source/repos/Plumbus/Plumbus/res/";

Window* window;
Clip* clip;
Timer* timer;
ColorWheel* colorwheel;
CudaInterface* GPU;


bool listen(int input) {
	switch (input) {
	case 'a':
		window->change_frame(-1);
		return true;

	case 'd':
		window->change_frame(1);
		return true;

	case 'q':
		return false;

	case 'k':
		window->increment_plane(-1);
		return true;
	case 'l':
		window->increment_plane(1);
		return true;
	default:
		return true;
	}
}

int main() {
	timer = new Timer();
	colorwheel = new ColorWheel();
	GPU = new CudaInterface();

	timer->begin("init clip");
	clip = new Clip(RES_PATH);
	timer->end("init clip");

	timer->generate_report();
	window = new Window(WINDOW_HEIGHT, WINDOW_WIDTH, clip);
	window->change_frame(0);


	//window->update_window("superpixels", window->frame()->superpixels());
	window->update_window("selective blur", window->frame()->blurred());
	window->update_window("regions", window->frame()->regions());
	//window->update_window("edge_overlay", window->frame()->edge_overlay());

	


	while (listen(cv::waitKey())) {


		//window->update_window("superpixels", window->frame()->superpixels());
		window->update_window("selective blur", window->frame()->blurred());
		window->update_window("regions", window->frame()->regions());
		//window->update_window("edge_overlay", window->frame()->edge_overlay());
		

	}
	return 0;
}

