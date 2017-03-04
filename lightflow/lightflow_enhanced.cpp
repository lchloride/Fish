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
cv::Ptr<BackgroundSubtractorMOG2> bgsubtractor;//�����������
const int ds = 0; //ʶ���ƫ�ƣ���֤��������ܹ���ʶ�����
int g_maxCornerNumber = 10; //��ȡ�ǵ�����ֵ��Ĭ��Ϊ2
const int r = 4;//���ƽǵ��Բ�İ뾶
const int thres = 13; //Ϊ��ʶ��ͷ�������е���ֵ����������ֵ(Ӧ����Ҫ��ѧϰ)
Point st, ed;
const int median_thres = 9;//��ֵ�˲��ĵ�λ���ڴ�С
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

	//��opencv3֮��������Ϊ�˳����࣬����ֱ�Ӵ�������
	bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setShadowThreshold(50);
	for (;;)
	{
		Mat frame;
		// ����Ƶ�в���һ֡
		cap >> frame;
		if (frame.empty())
			break;

		// ������Ƶ֡ΪͼƬ
		frame.copyTo(image);
		GaussianBlur(image, image, Size(5, 5), 0, 0);
		//medianBlur(image, image, median_thres);
		// ��ͼƬת�ɻҶ�ͼ
		cvtColor(image, gray, COLOR_BGR2GRAY);

		Mat dst = image.clone();
		Mat fgmask;
		getObject(image, image, fgmask);//�ӱ����л��ǰ��


		// ��ҹģʽ�����ǲ���ʾԭͼ
		if (nightMode)
			image = Scalar::all(0);

		// ��Ҫ����ȷ������������
		if (needToInit)
		{
			// automatic initialization
			// �ҽǵ�
			GoodFeaturesToTrack(gray, dst, fgmask, points[1]);
			if (points[0].size() == 0)
				seperateFishesByFeaturePt(fgmask, fish_num, points[1], points_index);
			else {
				matchArea(fgmask, points[0]);
				points[1].insert(points[1].end(), points[0].begin(), points[0].end());
			}
			//seperateHeadTail(gray, dst, points[1]);
			// �����ؼ��ǵ���
			//cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			needToInit = false;
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			// ������Ԥ��
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

				if (!status[i]) { // ��֡����һ֡��Ӧ�ĵ㶪ʧ
					cout << "Discard: " << i << endl;
					for (int j = getPointsIdx(points_index, i) + 1; j <= fish_num; j++)
						new_points_index[j]--;

					cout << "idx: ";
					for (int j = 0; j <= fish_num; j++)
						cout << new_points_index[j] << " ";
					cout << endl;
					
					circle(image, points[0][i], 3, Scalar(0, 0, 255), -1);//��һ֡��ʧ�ĵ�
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

//�ӱ����л��ǰ������Ĥ
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
	fgmask = Mat::zeros(frame.size(), CV_8UC1); //��Ĥ��ʼ��
	mask.copyTo(fgmask);
}

//��ʶ�����Ĥ��Ѱ������������Ĥ����Ϊ����������
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

//ʹ�ýǵ����㷨������������нǵ��⣬�õ���������Ϊͷβ��
//����ȡͷβ�����꣬������ԭ���ԭ��
void GoodFeaturesToTrack(const Mat src, Mat &dst, const Mat mask, vector<Point2f>& corners)
{
	//Shi-Tomasi�㷨��goodFeaturesToTrack�������Ĳ���׼��
	double qualityLevel = 0.1;//�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 15;//�ǵ�֮�����С����(�Ժ���Ҫ��ѧϰ)
	int blockSize = 7;//���㵼������ؾ���ʱָ��������Χ
	double k = 0.04;//Ȩ��ϵ��

					//����Shi-Tomasi�ǵ���
	goodFeaturesToTrack(src,//g_grayImage,//����ͼ��
		corners,//��⵽�Ľǵ���������
		g_maxCornerNumber,//�ǵ��������� 
		qualityLevel,//�ǵ���ɽ��ܵ���С����ֵ
		minDistance,//�ǵ�֮�����С����
		mask,//����Ȥ����
		blockSize,//���㵼������ؾ���ʱָ��������Χ
		false,//��ʹ��Harris�ǵ���
		k);//Ȩ��ϵ��
		   //���������Ϣ
		   //cout << "\t>�˴μ�⵽�Ľǵ�����Ϊ��" << corners.size() << endl;
	for (int i = 0; i < corners.size(); i++)
	{
		////���������ɫ���Ƴ��ǵ�
		//circle(dst, corners[i], r, Scalar(g_rng.uniform(0, 200), g_rng.uniform(0, 200),
		//	g_rng.uniform(0, 200)), -1, 8, 0);
		circle(dst, corners[i], r, Scalar(255, 0, 0), -1, 8, 0);
	}
}

//�ú�����ͷβ�������������ֿ�����ʹ����ֵ���ķ���
void seperateHeadTail(Mat src, Mat& dst, vector<Point2f> corners)
{
	vector<vector<Point> > contours; //���ʶ������������а��������������
	Mat binImage;//��Ŷ�ֵ����ͼ��

				 //��ֵ���ɺڰ�ͼ
	threshold(src, binImage, thres, 255, THRESH_BINARY);

	//�ںڰ�ͼ��Ѱ������
	vector<Vec4i> hierarchy;
	findContours(binImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (contours.size() == 0) //û�ҵ������򲻽��к�������
		return;

	Mat draw = Mat::zeros(binImage.size(), CV_8UC1);//draw��������ͼ�õ�
													//cout << "Area: ";
	double maxArea = 0; //��ֵ�����ͼ�ε�������
	int big_contour_num = 0; //��������Ӧ�������±�
	for (int i = 1; i < contours.size(); i++) //����ֵ����������������
	{
		//drawContours(draw, contours, i, Scalar(255, 0, 0));
		double area = contourArea(contours[i]);
		if (area > maxArea)
		{
			big_contour_num = i;
			maxArea = area;
		}
	}

	//���µ�һ��Ϊ�����ã��������ս��
	if (big_contour_num > 0)//���������������ͼ
		drawContours(draw, contours, big_contour_num, Scalar(0, 255, 0));

	//��������ǵ㵽������������̾��룬�����̾������Сֵ
	vector<double> point2contour; //��Žǵ㵽���������ľ���
	point2contour.assign(corners.size(), 0);
	int min_num = -1; //��������������Ľǵ���±�
	for (int i = 0; i < corners.size(); i++)
	{
		double t = pointPolygonTest(contours[big_contour_num], Point2f(corners[i].x, corners[i].y), true); //��㵽���������ľ���
		if (t >= 0) //���������ϻ��ڣ�������0
			point2contour[i] = 0;
		else //���������⣬������|t|
			point2contour[i] = -t;
		//point2contour[i] = -t;

		if (min_num == -1) //��������Сֵ
			min_num = 0;
		else
			if (point2contour[i] < point2contour[min_num])
				min_num = i;
	}
	//printf("min distance: %0.1lf\n", point2contour[min_num]);

	//����ֵд���ļ�
	if (min_num > -1)//���˵��ǵ�����Ϊ0�����
	{
		ed = Point(corners[min_num].x, corners[min_num].y);
			double v = sqrt(pow((ed.x - st.x), 2) + pow((ed.y - st.y), 2)) / (1.0 / 15);
			printf("v=%0.3lf\n", v);
		st = ed;
		circle(dst, corners[min_num], r, Scalar(255, 255, 255), -1, 8, 0);	 //�԰�ɫ���Ƴ�ͷ�ĵ㣬�õ�Ḳ��֮ǰ������ͷ����
	}
	//��ʾ�����£�ʶ��������
	imshow("dst", dst);

}

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar& color, int thickness, int lineType)
{
	const double PI = 3.1415926;
	Point arrow;
	//���� �� �ǣ���򵥵�һ�����������ͼʾ���Ѿ�չʾ���ؼ����� atan2 ��������������棩   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	//������Ǳߵ���һ�˵Ķ˵�λ�ã�����Ļ��������Ҫ����ͷ��ָ��Ҳ����pStart��pEnd��λ�ã� 
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
		count_points[max_idx] = -1;// ������ֵ���´�Ѱ��ʱ�������ҵ�
		//if (i + 1 < fish_num) // ��������������
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
	// �ҳ���������
	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	vector<int> points_min_idx;
	vector<double> points_min_dist;
	for (int i = 0; i < points.size(); i++) {
		int min_idx = 0;
		double min_dist = mask.cols + mask.rows;
		for (int j = 0; j < contours.size(); j++) {
			double d = pointPolygonTest(contours[j], points[i], true); //��㵽���������ľ���
			if (d >= 0)// �Ǹ�������ʾ�õ��ڸ�������/��
				d = 0;
			else // ������˵���õ��ڸ������⣬d�ľ���ֵ��Ϊ�õ���������Ĵ�ֱ����
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