#include <iostream>
#include <opencv\highgui.h>
#include <opencv\cv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\features2d.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("2.jpg");
	Mat descriptors;
	vector<KeyPoint> keypoints;
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;  
	params.maxThreshold = 100;  
	params.thresholdStep = 10;  
	params.minArea = 10;   
	params.minConvexity = 0.3;  
	params.minInertiaRatio = 0.01;  
	params.maxArea = 8000;  
	params.maxConvexity = 10;  
	params.filterByColor = false;  
	params.filterByCircularity = false;  
	SimpleBlobDetector blobDetector;
	cvtColor(image, image, COLOR_BGR2GRAY);
	blobDetector.create(params);
	blobDetector.detect(image, keypoints);
	drawKeypoints(image, keypoints, image, Scalar(255, 0, 0));
	return 0;
}