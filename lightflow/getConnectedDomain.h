#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stack>

using namespace std;
using namespace cv;
void getConnectedDomain(Mat& src, vector<Rect>& boundingbox);//boundingboxΪ���ս������Ÿ�����ͨ��İ�Χ��

