#include <opencv2/core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <chrono>

#include "CVModules/DustSeparationModule.h"
#include "CVModules/DustDensityEstimationModule.h"


cv::Mat image;
cv::Mat separatedDustImage;
cv::Mat densityEstimationImage;


DustSeparationModule DustSeparation_Module = DustSeparationModule(5);
DustDensityEstimationModule DustDensityEstimation_Module = DustDensityEstimationModule(50);

int main()
{
	std::vector<long long> elapsedTimes;
	elapsedTimes.reserve(30);
	elapsedTimes.clear();

	image = cv::imread("./TestImages/realWindow_1.jpg");
	DustDensityEstimation_Module.MemoryPreAllocate_GPU(image.rows, image.cols, 20, 36);


	for (int i = 0; i < 30; i++)
	{

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		DustSeparation_Module.SeparateDustFromImage(image, &separatedDustImage);

		bool isVisualize = false;
		float estimatedDensity = DustDensityEstimation_Module.EstimateDustDensity(
			separatedDustImage, &densityEstimationImage, 20, 36, isVisualize, DustDensityEstimationModule::TargetDevice::GPU);

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		cv::imshow("originalImage", image);
		cv::imwrite("originalImage.png", image);

		cv::imshow("separatedDustImage", separatedDustImage);

		cv::imshow("proposedAreas", densityEstimationImage);
		cv::imwrite("proposedAreas.png", densityEstimationImage);

		std::cout << estimatedDensity << std::endl;
		std::cout << i <<"/Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

		cv::waitKey(1);
		
		elapsedTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
	}

	double meanElapsedTime= 0;
	for (auto it = elapsedTimes.begin(); it < elapsedTimes.end(); it++)
	{
		meanElapsedTime += (double)(*it);
	}

	meanElapsedTime = meanElapsedTime / 30.0;

	std::cout << "mean Elapsed Time(ms)" << meanElapsedTime << std::endl;


	

	cv::waitKey(0);
}
