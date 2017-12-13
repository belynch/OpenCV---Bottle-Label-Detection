#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

#define NO_IMAGES 30
#define NO_CHANNELS 3
#define HUE 0
#define LUMINANCE 1
#define SATURATION 2

Scalar standard_deviations[NO_IMAGES];
bool groundTruth[NO_IMAGES];
bool results[NO_IMAGES];
int FP, FN, TP, TN = 0;
double precision, recall, accuracy, specificity, f1 = 0.0;

/**
 *
 * Initalises the groundTruth array
 *
 */
void initialiseArray(){
	for(int i = 0; i < NO_IMAGES; i++){
		groundTruth[i] = true;
	}

	//unlabeled bottles receive the value false
	groundTruth[4] = false;
	groundTruth[6] = false;
	groundTruth[13] = false;
	groundTruth[16] = false;
	groundTruth[17] = false;
	groundTruth[20] = false;
	groundTruth[23] = false;
	groundTruth[29] = false;
}

/**
 *
 * Function taken from chapter 8.6.3 of 'A Practical 
 * Introduction to Computer Vision with OpenCV' by
 * Kenneth Dawson-Howe
 *
 */
void performanceTest(){
	for (int i=0; i < NO_IMAGES; i++){
		bool result = results[i];
		bool gt = groundTruth[i];

		if(gt)
			if (result)
				TP++;
			else FN++;
		else if (result)
			FP++;
		else TN++;
	}

	precision = ((double) TP) / ((double) (TP+FP));
	recall = ((double) TP) / ((double) (TP+FN));
	accuracy = ((double) (TP+TN)) / ((double) (TP+FP+TN+FN));
	specificity = ((double) TN) / ((double) (FP+TN));
	f1 = 2.0*precision*recall / (precision + recall);
}

int main(int argc, const char** argv)
{
	#pragma region LOAD_IMAGES
	string image_files[] = {
		"Media/glue-0.png",
		"Media/glue-1.png",
		"Media/glue-2.png",
		"Media/glue-3.png",
		"Media/glue-4.png",
		"Media/glue-5.png",
		"Media/glue-6.png",
		"Media/glue-7.png",
		"Media/glue-8.png",
		"Media/glue-9.png",
		"Media/glue-10.png",
		"Media/glue-11.png",
		"Media/glue-12.png",
		"Media/glue-13.png",
		"Media/glue-14.png",
		"Media/glue-15.png",
		"Media/glue-16.png",
		"Media/glue-17.png",
		"Media/glue-18.png",
		"Media/glue-19.png",
		"Media/glue-20.png",
		"Media/glue-21.png",
		"Media/glue-22.png",
		"Media/glue-23.png",
		"Media/glue-24.png",
		"Media/glue-25.png",
		"Media/glue-26.png",
		"Media/glue-27.png",
		"Media/glue-28.png",
		"Media/glue-29.png",
	};

	//Load images into an array 
	Mat* images = new Mat[NO_IMAGES];
	for(int i = 0; i < NO_IMAGES; i++){
		images[i] = imread(image_files[i], CV_LOAD_IMAGE_COLOR);
	}
	#pragma endregion LOAD_IMAGES
	
	initialiseArray();
	
	//Iterate through the array of images
	for(int i =0; i < NO_IMAGES; i++){
		//Crop image to remove background and lid
		Rect myROI(35, 180, 50, 60);
		Mat croppedImage = images[i](myROI);
		//Convert cropped image to HSL
		Mat src;
		Mat hsl;
		cvtColor(croppedImage, hsl, CV_RGB2HLS);
		//Separate HSL image into three channels
		Mat hslChannels[NO_CHANNELS];
		split(hsl, hslChannels);
		//Calculate standard deviation of the saturation channel
		Scalar mean, stddev;
		meanStdDev(hslChannels[SATURATION], mean, stddev);
		standard_deviations[i] = stddev;
		//imshow(to_string(i), hslChannels[SATURATION]); 
	}

	//Select test bottles without labels and identify threshold
	double bottle13 = standard_deviations[13][0];
	double bottle17 = standard_deviations[17][0];
	double threshold = max(bottle13,bottle17);

	//Outputs the numbers of labeled bottles 
	for(int i = 0; i < NO_IMAGES; i++){
		//if its sd is greater than the threshold it has a label
		if(standard_deviations[i][0] > threshold){
			cout<<"labeled bottle: "<<i<< endl;
			results[i] = true;
		}
	}
	
	//Calculate metrics
	performanceTest();
	cout << "False Positives: " << FP << endl;
	cout << "False Negatives: " << FN  << endl;
	cout << "True Positives: " << TP << endl;
	cout << "True Negatives: " << TN << endl;
	cout << "precision: " << precision << endl;
	cout << "recall: " << recall << endl;
	cout << "accuracy: " << accuracy << endl;
	cout << "specificity: " << specificity << endl;
	cout << "f1: " << f1 << endl;

	waitKey(0);
}


