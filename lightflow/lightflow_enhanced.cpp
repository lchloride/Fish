#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "selectionsort.h"
#include "areaStatistic.h"

#include <iostream>
#include <ctype.h>
#include <ctime>

using namespace cv;
using namespace std;

void refineSegments(const Mat& img, Mat& mask);
void getObject(const Mat frame, Mat &dst, Mat &fgmask);
void GoodFeaturesToTrack(const Mat src, const Mat mask, vector<Point2f>& corners);
void seperateHeadTail(Mat src, Mat& dst, vector<Point2f> corners);
void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar& color, int thickness=1, int lineType=8);
void seperateFishesByFeaturePt(Mat mask, int fish_num, vector<Point2f>& feature_points,
	vector<int> &feature_points_index);
int getPointsIdx(vector<int> idx, int x);
bool checkFeaturePoints(vector<int> points_index);
void matchArea(Mat gray, Mat mask, vector<Point2f>& points, vector<int>& points_index, Mat& dst);


static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nʹ�� Lucas Kanade �㷨׷�ٶ�����\n"
		"Using OpenCV version " << CV_VERSION << endl;
	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tn - switch the \"night\" mode on/off\n"<< endl;
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
bool halt = false; // ǿ���˳���־
Scalar contour_color[] = { Scalar(0, 165, 255), Scalar(255, 191, 0) };
AreaStatistic area_statistic;
const int init_time = 2;// ��ʼ����ʱ��

int main(int argc, char** argv)
{
	help();

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(41, 41);

	const int MAX_COUNT = 5;
	bool needToInit = true;
	bool nightMode = false;
	bool updateFeaturePoint = true;
	
	int frame_count = 0;

	VideoCapture cap("0204.avi");

	if (!cap.isOpened())
	{
		cout << "Could not open video...\n";
		return 0;
	}

	int FRAME_RATIO = (int)cap.get(CV_CAP_PROP_FPS); // ֡��
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

		frame_count++;

		// ������Ƶ֡ΪͼƬ
		frame.copyTo(image);
		GaussianBlur(image, image, Size(5, 5), 0, 0);
		// ��ͼƬת�ɻҶ�ͼ
		cvtColor(image, gray, COLOR_BGR2GRAY);

		Mat dst = image.clone();
		Mat fgmask;
		getObject(image, image, fgmask);//�ӱ����л��ǰ��


		// ��ҹģʽ�����ǲ���ʾԭͼ
		if (nightMode)
			image = Scalar::all(0);

		// ��Ҫ����ȷ������������
		if (frame_count > init_time*FRAME_RATIO && needToInit)
		{
			// �ҽǵ�
			GoodFeaturesToTrack(gray, fgmask, points[1]);
			seperateFishesByFeaturePt(fgmask, fish_num, points[1], points_index);
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
				if (!status[i]) { // ��֡����һ֡��Ӧ�ĵ㶪ʧ
					cout << "Discard: " << i << endl;
					for (int j = getPointsIdx(points_index, i) + 1; j <= fish_num; j++)
						new_points_index[j]--;

					cout << "idx: ";
					for (int j = 0; j <= fish_num; j++)
						cout << new_points_index[j] << " ";
					cout << endl;
					
					circle(dst, points[0][i], 3, Scalar(0, 0, 255), -1);//��һ֡��ʧ�ĵ�
					continue;
				}

				circle(dst, points[0][i], 3, Scalar(0, 255, 0), -1);//line end
				circle(dst, points[1][i], 3, Scalar(255, 0, 0), -1);//line start
				drawArrow(dst, points[0][i], points[1][i], 6, 30, Scalar(0, 255, 255));//an arrow from start to end

				points[1][k++] = points[1][i];

			}
			points[1].resize(k);
			points_index.assign(new_points_index.begin(), new_points_index.end());

			for (int j = 0; j < fish_num; j++) {
				cout << "Fish " << j << ": ";
				if (points_index[j] == points_index[j + 1]) {
					cout << "Target is lost" << endl;
					halt = true;
				}
				for (int h = points_index[j]; h < points_index[j + 1]; h++)
					cout << "(" << points[1][h].x << ", " << points[1][h].y << ") ";
				cout << endl;
			}

			matchArea(gray, fgmask, points[1], points_index, dst);
		}

		imshow("Tracking Fishes", dst);
		

		char c = (char)waitKey(10);
		if (halt || c == 27)
			break;
		switch (c)
		{
		case 'r':
			//needToInit = true;
			break;
		case 'c':
			//points[0].clear();
			//points[1].clear();
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

	bgsubtractor->apply(frame, mask, update_bg_model ? -1 : 0);

	Mat fgimg;
	fgimg = Scalar::all(0);
	frame.copyTo(fgimg, mask);
	refineSegments(frame, mask);
	fgmask = Mat::zeros(frame.size(), CV_8UC1); //��Ĥ��ʼ��
	mask.copyTo(fgmask);
}

//��ʶ�����Ĥ��Ѱ������������Ĥ����Ϊ����������
void refineSegments(const Mat& img, Mat& mask)
{
	int niters = 1;
	Mat temp;

	temp = mask;
	dilate(mask, temp, Mat(3, 3, CV_8U), Point(-1, -1), niters);
	erode(temp, temp, Mat(3, 3, CV_8U), Point(-1, -1), niters+1);
	dilate(temp, temp, Mat(3,3,CV_8U), Point(-1, -1), niters+1);
	temp.copyTo(mask);
	

}

//ʹ�ýǵ����㷨������������нǵ��⣬�õ���������Ϊͷβ��
//����ȡͷβ�����꣬������ԭ���ԭ��
void GoodFeaturesToTrack(const Mat src, const Mat mask, vector<Point2f>& corners)
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
		area_statistic.pushArea(contourArea(contours[max_idx])); // ͳ��ÿ���������ľ�ֵ�ͷ���

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

void matchArea(Mat gray, Mat mask, vector<Point2f>& points, vector<int>& points_index, Mat& dst) {
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
	//for (int i = 0; i < points_min_dist.size(); i++)
	//	cout << points_min_idx[i] << ", " << points_min_dist[i] << endl;
	//cout << endl;

	vector<int> match(points_index.size() - 1, -1); // match�洢����һ֡����һ֡ÿ�����ӳ���ϵ
	for (int i = 0; i < points_index.size() - 1; i++) {
		// ͳ����̾�����ֵĴ�����Ҳ���������ÿ�������п��ܶ�Ӧ����һ֡���ĸ�������
		vector<int> count_min_dist(contours.size(), 0);
		for (int j = points_index[i]; j < points_index[i+1]; j++)
			count_min_dist[points_min_idx[j]]++;
		// �ҳ���̾�����ֵĴ����е����ֵ����������������п��ܶ�Ӧ����(��)������
		int max_count = 0; // �洢�����ֵ
		int max_count_times = 0; // �洢�����ֵ���ֵĴ����������Ϊ1
		for (int j = 0; j < count_min_dist.size(); j++)
			if (count_min_dist[j] > max_count) {
				max_count = count_min_dist[j];
				max_count_times = 1;
			}
			else if (count_min_dist[j] == max_count)
				max_count_times++;

		if (max_count_times == 1) {
			// �洢���������µ�һ֡�е�������
			for (int j = 0; j < count_min_dist.size(); j++)
				if (count_min_dist[j] == max_count) {
					match[i] = j;
					break;
				}
		}
		else {
			// �ж�����򶼿�����������������Ƶ֡�еĶ�Ӧ
			Mat err = Mat::zeros(mask.size(), CV_8UC3);

			for (int j = 0; j < contours.size(); j++) {
				drawContours(err, contours, j, Scalar(200, 0, 120), CV_FILLED, 8, hierarchy);
			}
			bool decided = false;
			for (int j = 0; j < count_min_dist.size(); j++)
				if (count_min_dist[j] == max_count) {
					drawContours(err, contours, j, Scalar(130, 25, 25), CV_FILLED, 8, hierarchy);
					double area = contourArea(contours[j]);
					cout << "Matching area decision: " << j << ", area: " << area << ", " << area_statistic.checkRange(area) << endl;
					if (area_statistic.checkRange(area)) {
						//������������Ƕ�����һ����ĸ��ʺܴ�
						match[i] = j;
						decided = true;
						break;
					}
				}
			if (!decided) {
				int max_idx = 0;
				int max_area = 0;
				for (int j = 0; j < count_min_dist.size(); j++)
					if (count_min_dist[j] == max_count) {
						double d = contourArea(contours[j]);
						if (d > max_area) {
							max_area = d;
							max_idx = j;
						}
					}
				cout << "Max area: " << max_area << ", index: " << max_idx << endl;
				match[i] = max_idx;
				decided = true;
			}

			for (int j = 0; j < points.size(); j++)
				circle(err, points[j], 3, Scalar(120,255, 0));
			char filename[100] = { 0 };
			time_t rawtime;
			struct tm * timeinfo;
			time(&rawtime);
			timeinfo = localtime(&rawtime);
			if (decided)
				strftime(filename, 100, ".\\error\\error_decided_%H_%M_%S.png", timeinfo);
			else
				strftime(filename, 100, ".\\error\\error_undecided_%H_%M_%S.png", timeinfo);
			imwrite(filename, err);
			if (!decided) {
				halt = true;
				return;
			}
		}
	}

	// �Ѿ���ȡ��ÿ��������һ֡����һ֡�Ķ�Ӧ��ϵ������match��
	// �̶������������Ƿ����ص�
	bool overlap = false;
	for (int i = 0; i < points_index.size() - 1; i++) {
		for (int j = i+1; j < points_index.size() - 1; j++) {
			if (match[i] == match[j]) {
				overlap = true;
				break;
			}
		}
	}
	if (overlap) {
		cout << "Overlap!" << endl;
		for (int i = 0; i < match.size(); i++)
			drawContours(dst, contours, match[i], contour_color[i], 2);
		return;
	}
	else {
		for (int i = 0; i < match.size(); i++)
			area_statistic.pushArea(contourArea(contours[match[i]]));
		cout << "avg:" << area_statistic.getAvg() << ", var:" << area_statistic.getVar() << endl;

		// ������һ֡�ĵ㣬ɾ������������ĵ�
		;// ֮������

		// Ѱ�ұ�֡�Ľǵ㣬��Ķ�Ӧ���ֻᱻ����
		vector<Point2f> new_feature_points;
		GoodFeaturesToTrack(gray, mask, new_feature_points);
		for (int i = 0; i < points.size(); i++)
			new_feature_points.push_back(points[i]);

		vector<int>position(new_feature_points.size(), -1);
		// ���ÿ���µ����������ڵ�����
		for (int i = 0; i < new_feature_points.size(); i++) {
			for (int j = 0; j < contours.size(); j++) {
				double d = pointPolygonTest(contours[j], new_feature_points[i], true); //
				if (d >= 0) {
					position[i] = j;
					break;
				}
			}
		}

		points.clear();
		for (int i = 0; i < match.size(); i++) {
			for (int j = 0; j < new_feature_points.size(); j++)
				if (position[j] == match[i]) {
					points.push_back(new_feature_points[j]);
				}
			points_index[i + 1] = points.size();
			drawContours(dst, contours, match[i], contour_color[i], 2);
		}
	}

}