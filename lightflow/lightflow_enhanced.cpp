#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "selectionsort.h"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

void refineSegments(const Mat& img, Mat& mask);
void getObject(const Mat frame, Mat &dst, Mat &fgmask);
void GoodFeaturesToTrack(const Mat src, Mat &dst, const Mat mask, vector<Point2f>& corners);
void seperateHeadTail(Mat src, Mat& dst, vector<Point2f> corners);
void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar& color, int thickness=1, int lineType=8);
void seperateFishesByFeaturePt(Mat mask, int fish_num, vector<Point2f>& feature_points,
	vector<int> &feature_points_index);
int getPointsIdx(vector<int> idx, int x);
bool checkFeaturePoints(vector<int> points_index);
void matchArea(Mat mask, vector<Point2f> points);

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;
cv::Ptr<BackgroundSubtractorMOG2> bgsubtractor;//背景处理变量
const int ds = 0; //识别框偏移，保证鱼的轮廓能够在识别框中
int g_maxCornerNumber = 10; //获取角点的最大值，默认为2
const int r = 4;//绘制角点的圆的半径
const int thres = 13; //为了识别头部而进行的阈值化操作的阈值(应该需要自学习)
Point st, ed;
const int median_thres = 9;//中值滤波的单位窗口大小
int fish_num = 2;

//static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
//{
//	if (event == CV_EVENT_LBUTTONDOWN)
//	{
//		point = Point2f((float)x, (float)y);
//		addRemovePt = true;
//	}
//}

int main(int argc, char** argv)
{
	help();

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 5;
	bool needToInit = false;
	bool nightMode = false;
	bool updateFeaturePoint = true;

	VideoCapture cap("0204.avi");

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	namedWindow("LK Demo", 1);
	//setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image;
	vector<Point2f> points[2];
	vector<int> points_index;

	//在opencv3之后，这个类成为了抽象类，不能直接创建对象
	bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setShadowThreshold(50);
	for (;;)
	{
		Mat frame;
		// 从视频中捕获一帧
		cap >> frame;
		if (frame.empty())
			break;

		// 备份视频帧为图片
		frame.copyTo(image);
		GaussianBlur(image, image, Size(5, 5), 0, 0);
		//medianBlur(image, image, median_thres);
		// 将图片转成灰度图
		cvtColor(image, gray, COLOR_BGR2GRAY);

		Mat dst = image.clone();
		Mat fgmask;
		getObject(image, image, fgmask);//从背景中获得前景


		// 黑夜模式，就是不显示原图
		if (nightMode)
			image = Scalar::all(0);

		// 需要重新确定特征点的情况
		if (needToInit)
		{
			// automatic initialization
			// 找角点
			GoodFeaturesToTrack(gray, dst, fgmask, points[1]);
			if (points[0].size() == 0)
				seperateFishesByFeaturePt(fgmask, fish_num, points[1], points_index);
			else {
				matchArea(fgmask, points[0]);
				points[1].insert(points[1].end(), points[0].begin(), points[0].end());
			}
			//seperateHeadTail(gray, dst, points[1]);
			// 亚像素级角点检测
			//cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			needToInit = false;
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			// 光流法预测
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			vector<int> new_points_index(points_index.begin(), points_index.end());

			for (i = k = 0; i < points[1].size(); i++)
			{
			//	if (addRemovePt)
			//	{
			//		if (norm(point - points[1][i]) <= 5)
			//		{
			//			addRemovePt = false;
			//			continue;
			//		}
			//	}

				if (!status[i]) { // 本帧中上一帧对应的点丢失
					cout << "Discard: " << i << endl;
					for (int j = getPointsIdx(points_index, i) + 1; j <= fish_num; j++)
						new_points_index[j]--;

					cout << "idx: ";
					for (int j = 0; j <= fish_num; j++)
						cout << new_points_index[j] << " ";
					cout << endl;
					
					circle(image, points[0][i], 3, Scalar(0, 0, 255), -1);//上一帧丢失的点
					continue;
				}

				circle(image, points[0][i], 3, Scalar(0, 255, 0), -1);//line end
				circle(image, points[1][i], 3, Scalar(255, 0, 0), -1);//line start
				drawArrow(image, points[0][i], points[1][i], 6, 30, Scalar(0, 255, 255));//an arrow from start to end

				points[1][k++] = points[1][i];
				//cout << "[" << err[i] <<"] ";

			}
			points[1].resize(k);
			points_index.assign(new_points_index.begin(), new_points_index.end());

			for (int j = 0; j < fish_num; j++) {
				cout << "Fish " << j << ": ";
				for (int h = points_index[j]; h < points_index[j + 1]; h++)
					cout << "(" << points[1][h].x << ", " << points[1][h].y << ") ";
				cout << endl;
			}
			if (checkFeaturePoints(points_index))
				needToInit = true;
		}

		//if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		//{
		//	vector<Point2f> tmp;
		//	tmp.push_back(point);
		//	cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
		//	points[1].push_back(tmp[0]);
		//	addRemovePt = false;
		//}

		//needToInit = false;
		imshow("LK Demo", image);
		

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}

	return 0;
}

//从背景中获得前景的掩膜
void getObject(const Mat frame, Mat &dst, Mat &fgmask)
{
	bool update_bg_model = true;

	vector<vector<Point> > contours;
	Mat mask;
	//fgmask.release();

	bgsubtractor->apply(frame, mask, update_bg_model ? -1 : 0);

	Mat fgimg;
	fgimg = Scalar::all(0);
	frame.copyTo(fgimg, mask);
	//imshow("fgimg", fgimg);
	imwrite("mask.jpg", mask);
	//imwrite("src.jpg", g_srcImage);
	refineSegments(frame, mask);
	fgmask = Mat::zeros(frame.size(), CV_8UC1); //掩膜初始化
	mask.copyTo(fgmask);
}

//从识别的掩膜中寻找最大面积的掩膜，作为鱼所在区域
void refineSegments(const Mat& img, Mat& mask)
{
	int niters = 1;

	Mat temp;

	//contours.clear();
	temp = mask;
	dilate(mask, temp, Mat(3, 3, CV_8U), Point(-1, -1), niters);
	erode(temp, temp, Mat(3, 3, CV_8U), Point(-1, -1), niters+1);
	dilate(temp, temp, Mat(4,4,CV_8U), Point(-1, -1), niters+1);
	temp.copyTo(mask);
	

}

//使用角点检测算法对鱼的轮廓进行角点检测，得到的特征点为头尾，
//并提取头尾的坐标，以坐标原点的原点
void GoodFeaturesToTrack(const Mat src, Mat &dst, const Mat mask, vector<Point2f>& corners)
{
	//Shi-Tomasi算法（goodFeaturesToTrack函数）的参数准备
	double qualityLevel = 0.1;//角点检测可接受的最小特征值
	double minDistance = 15;//角点之间的最小距离(以后需要自学习)
	int blockSize = 7;//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;//权重系数

					//进行Shi-Tomasi角点检测
	goodFeaturesToTrack(src,//g_grayImage,//输入图像
		corners,//检测到的角点的输出向量
		g_maxCornerNumber,//角点的最大数量 
		qualityLevel,//角点检测可接受的最小特征值
		minDistance,//角点之间的最小距离
		mask,//感兴趣区域
		blockSize,//计算导数自相关矩阵时指定的邻域范围
		false,//不使用Harris角点检测
		k);//权重系数
		   //输出文字信息
		   //cout << "\t>此次检测到的角点数量为：" << corners.size() << endl;
	for (int i = 0; i < corners.size(); i++)
	{
		////以随机的颜色绘制出角点
		//circle(dst, corners[i], r, Scalar(g_rng.uniform(0, 200), g_rng.uniform(0, 200),
		//	g_rng.uniform(0, 200)), -1, 8, 0);
		circle(dst, corners[i], r, Scalar(255, 0, 0), -1, 8, 0);
	}
}

//该函数将头尾两个特征点区分开来，使用阈值化的方法
void seperateHeadTail(Mat src, Mat& dst, vector<Point2f> corners)
{
	vector<vector<Point> > contours; //存放识别的轮廓，其中包含鱼的躯干轮廓
	Mat binImage;//存放二值化的图像

				 //阈值化成黑白图
	threshold(src, binImage, thres, 255, THRESH_BINARY);

	//在黑白图中寻找轮廓
	vector<Vec4i> hierarchy;
	findContours(binImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (contours.size() == 0) //没找到轮廓则不进行后续操作
		return;

	Mat draw = Mat::zeros(binImage.size(), CV_8UC1);//draw是用来绘图用的
													//cout << "Area: ";
	double maxArea = 0; //阈值化后的图形的最大面积
	int big_contour_num = 0; //最大面积对应的轮廓下标
	for (int i = 1; i < contours.size(); i++) //求阈值化后轮廓的最大面积
	{
		//drawContours(draw, contours, i, Scalar(255, 0, 0));
		double area = contourArea(contours[i]);
		if (area > maxArea)
		{
			big_contour_num = i;
			maxArea = area;
		}
	}

	//以下的一句为调试用，并非最终结果
	if (big_contour_num > 0)//已求出最大面积，绘图
		drawContours(draw, contours, big_contour_num, Scalar(0, 255, 0));

	//求出两个角点到躯干轮廓的最短距离，求出最短距离的最小值
	vector<double> point2contour; //存放角点到躯干轮廓的距离
	point2contour.assign(corners.size(), 0);
	int min_num = -1; //到躯干轮廓最近的角点的下标
	for (int i = 0; i < corners.size(); i++)
	{
		double t = pointPolygonTest(contours[big_contour_num], Point2f(corners[i].x, corners[i].y), true); //求点到躯干轮廓的距离
		if (t >= 0) //点在轮廓上或内，距离是0
			point2contour[i] = 0;
		else //点在轮廓外，距离是|t|
			point2contour[i] = -t;
		//point2contour[i] = -t;

		if (min_num == -1) //求距离的最小值
			min_num = 0;
		else
			if (point2contour[i] < point2contour[min_num])
				min_num = i;
	}
	//printf("min distance: %0.1lf\n", point2contour[min_num]);

	//将数值写入文件
	if (min_num > -1)//过滤掉角点数量为0的情况
	{
		ed = Point(corners[min_num].x, corners[min_num].y);
			double v = sqrt(pow((ed.x - st.x), 2) + pow((ed.y - st.y), 2)) / (1.0 / 15);
			printf("v=%0.3lf\n", v);
		st = ed;
		circle(dst, corners[min_num], r, Scalar(255, 255, 255), -1, 8, 0);	 //以白色绘制出头的点，该点会覆盖之前画出的头部点
	}
	//显示（更新）识别结果窗口
	imshow("dst", dst);

}

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar& color, int thickness, int lineType)
{
	const double PI = 3.1415926;
	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
}

void seperateFishesByFeaturePt(Mat mask, int fish_num, vector<Point2f>& feature_points, 
	vector<int> &feature_points_index) 
{
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	Mat dst = Mat::zeros(mask.size(), CV_8UC3);

	if (contours.size() < fish_num)
		return;
	//if (fish_num == 0)
	//	fish_num = contours.size();

	Mat area;
	vector<int> count_points(contours.size(), 0);

	for (int i = 0; i < contours.size(); i++) {
		area = Mat::zeros(mask.size(), CV_8UC1);
		drawContours(area, contours, i, Scalar(255), CV_FILLED, 8, hierarchy);
		drawContours(dst, contours, i, Scalar(200, 0, 120), CV_FILLED, 8, hierarchy);
		for (int j = 0; j < feature_points.size(); j++) {
			if (area.at<uchar>(feature_points[j]) != 0) {
				count_points[i]++;
			}
		}
		char str[100] = { 0 };
		sprintf(str, "mask_%d.png", i);
		imwrite(str, area);
	}
	for (int i = 0; i < count_points.size(); i++)
		cout << count_points[i] << " ";
	cout << endl;

	vector<Point2f> refined_feature_points;
	feature_points_index.assign(fish_num+1, 0);

	for (int i = 0; i < fish_num; i++) {
		int max_idx = 0;
		for (int j = 1; j < count_points.size(); j++)
			if (count_points[max_idx] < count_points[j])
				max_idx = j;
		int max_count = count_points[max_idx];
		count_points[max_idx] = -1;// 清除最大值，下次寻找时将不会找到
		//if (i + 1 < fish_num) // 更新特征点索引
			feature_points_index[i + 1] = feature_points_index[i] + max_count;

		area = Mat::zeros(mask.size(), CV_8UC1);
		drawContours(area, contours, max_idx, Scalar(255), CV_FILLED, 8, hierarchy);
		drawContours(dst, contours, max_idx, Scalar(0,0,120), CV_FILLED, 8, hierarchy);

		for (int j = 0; j < feature_points.size(); j++) {
			if (area.at<uchar>(feature_points[j]) != 0) {
				refined_feature_points.push_back(feature_points[j]);
			}
		}

	}
	for (int i = 0; i < feature_points.size(); i++)
		circle(dst, feature_points[i], r, Scalar(255, 255, 0), -1, 8, 0);
	feature_points.assign(refined_feature_points.begin(), refined_feature_points.end());
	for (int i = 0; i < feature_points.size(); i++) 
		circle(dst, feature_points[i], r, Scalar(255, 0, 0), -1, 8, 0);
	imshow("fgimg", dst);

}

void modifyFeaturePoints() {

}

int getPointsIdx(vector<int> idx, int x) {
	for (int i = 0; i < fish_num; i++)
		if (x >= idx[i] && x < idx[i + 1])
			return i;
}

bool checkFeaturePoints(vector<int> points_index) {
	for (int i = 0; i < fish_num; i++)
		if (points_index[i + 1] - points_index[i] <= 1)
			return true;
	return false;
}

void matchArea(Mat mask, vector<Point2f> points) {
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	// 找出所有轮廓
	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	vector<int> points_min_idx;
	vector<double> points_min_dist;
	for (int i = 0; i < points.size(); i++) {
		int min_idx = 0;
		double min_dist = mask.cols + mask.rows;
		for (int j = 0; j < contours.size(); j++) {
			double d = pointPolygonTest(contours[j], points[i], true); //求点到躯干轮廓的距离
			if (d >= 0)// 非负数，表示该点在该区域中/上
				d = 0;
			else // 负数，说明该点在该区域外，d的绝对值作为该点距离该区域的垂直距离
				d = fabs(d);
			if (d < min_dist) {
				min_dist = d;
				min_idx = j;
			}
		}
		points_min_idx.push_back(min_idx);
		points_min_dist.push_back(min_dist);
	}
	cout << "points_min_dist" << endl;
	for (int i = 0; i < points_min_dist.size(); i++)
		cout << points_min_idx[i] << ", " << points_min_dist[i] << endl;
}