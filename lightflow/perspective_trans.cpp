#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("1.2_1.jpg");
	int img_height = img.rows;
	int img_width = img.cols;
	vector<Point2f> corners(4);
	corners[0] = Point2f(29, 162);
	corners[1] = Point2f(29, 421);
	corners[2] = Point2f(647, 422);
	corners[3] = Point2f(647, 162);
	vector<Point2f> corners_trans(4);
	corners_trans[0] = Point2f(15, 163);
	corners_trans[1] = Point2f(18, 439);
	corners_trans[2] = Point2f(677, 422);
	corners_trans[3] = Point2f(674, 164);

	Mat transform = getPerspectiveTransform(corners, corners_trans);
	cout << transform << endl;
	//cv::warpPerspective(im, quad, transmtx, quad.size());
	Mat img_trans = Mat::zeros(img_height, img_width, CV_8UC3);
	warpPerspective(img, img_trans, transform, img_trans.size());
	Rect rect(15, 163, (679 - 18), (439-163));
	Mat ROI = img_trans(rect);
	img_trans = ROI.clone();
	imshow("quadrilateral", img_trans);
	int c = waitKey();
	imwrite("result.png", img_trans);
	return  0;
}