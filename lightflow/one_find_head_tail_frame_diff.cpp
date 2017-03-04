//--------------------------------------������˵����-------------------------------------------
//		����˵������OpenCV3������š�OpenCV3���鱾����ʾ������87
//		����������Shi-Tomasi�ǵ���ʾ��
//		�����������ò���ϵͳ�� Windows 7 64bit
//		������������IDE�汾��Visual Studio 2010
//		������������OpenCV�汾��	3.0 beta
//		2014��11�� Created by @ǳī_ë����
//		2014��12�� Revised by @ǳī_ë����
//------------------------------------------------------------------------------------------------



//---------------------------------��ͷ�ļ��������ռ�������֡�----------------------------
//		����������������ʹ�õ�ͷ�ļ��������ռ�
//------------------------------------------------------------------------------------------------
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "3.h"
#include <iostream>
#include <cstdio>
#include <ctime>
using namespace cv;
using namespace std;

//-----------------------------------���궨�岿�֡�-------------------------------------------- 
//  ����������һЩ������ 
//----------------------------------------------------------------------------------------------
#define WINDOW_NAME "��Shi-Tomasi�ǵ��⡿"        //Ϊ���ڱ��ⶨ��ĺ� 
#define threshold_diff1 8 //���ü�֡���ֵ
#define threshold_diff2 8 //���ü�֡���ֵ



//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//          ������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_grayImage, g_binImage, last2, last1;
int g_maxCornerNumber = 33;
int g_maxTrackbarNumber = 500;
RNG g_rng(12345);//��ʼ�������������
//double dist = 0;
//Point2f lastpoint;
int framenum = 0;
//const double frametime = 0.0666666666666667;
const int thres = 20;
vector<vector<Point> > contours;
int contour_num=1;
Mat mask;
int ds = 20;
int counter = 0;
Mat vertex;
Mat last2_color, last1_color;
void frame_diff(Mat img_src1, Mat img_src2, Mat img_src3);

//-----------------------------��on_GoodFeaturesToTrack( )������----------------------------
//          ��������Ӧ�������ƶ���Ϣ�Ļص�����
//----------------------------------------------------------------------------------------------
void on_GoodFeaturesToTrack( int, void* )
{
	//��1���Ա���С�ڵ���1ʱ�Ĵ���
	if( g_maxCornerNumber <= 1 ) { g_maxCornerNumber = 1; }
	g_maxCornerNumber = 3;
	//��2��Shi-Tomasi�㷨��goodFeaturesToTrack�������Ĳ���׼��
	vector<Point2f> corners;
	double qualityLevel = 0.01;//�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 50;//�ǵ�֮�����С����
	int blockSize = 7;//���㵼������ؾ���ʱָ��������Χ
	double k = 0.04;//Ȩ��ϵ��
	Mat copy = last2.clone();	//����Դͼ��һ����ʱ�����У���Ϊ����Ȥ����
	//Mat copy = g_grayImage.clone();	//����Դͼ��һ����ʱ�����У���Ϊ����Ȥ����

	//Point* rect_vertex=NULL;
	//int count = 0;
	cvtColor(copy, copy, COLOR_GRAY2BGR);
	frame_diff(last2, last1, g_grayImage);
	vector<vector<Point> > frame;
	findContours(vertex, frame, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (frame.size() > 0)
		drawContours(copy, frame, 0, Scalar(255, 255, 255), 1, 8);
	else
	{
		printf("no contours\n");
		return;
	}
	//��3������Shi-Tomasi�ǵ���
	goodFeaturesToTrack( last2,//����ͼ��
		corners,//��⵽�Ľǵ���������
		g_maxCornerNumber,//�ǵ���������
		qualityLevel,//�ǵ���ɽ��ܵ���С����ֵ
		minDistance,//�ǵ�֮�����С����
		Mat(),//����Ȥ����
		blockSize,//���㵼������ؾ���ʱָ��������Χ
		false,//��ʹ��Harris�ǵ���
		k );//Ȩ��ϵ��


	//��4�����������Ϣ
	cout<<"\t>�˴μ�⵽�Ľǵ�����Ϊ��"<<corners.size()<<endl;

	//��5�����Ƽ�⵽�Ľǵ�
	int r = 4;
	FILE *fp;
	fp = fopen("result.txt", "a");
	Point2f midpoint;
	vector<double> point2contour;
	point2contour.assign(corners.size(), 0);
	vector<double> tail2contour;
	int min_num=-1;
	for( int i = 0; i < corners.size(); i++ )
	{ 
		//���������ɫ���Ƴ��ǵ�
		circle( copy, corners[i], r, Scalar(g_rng.uniform(0,200), g_rng.uniform(100,200),
			g_rng.uniform(100,200)), -1, 8, 0 ); 
		//fprintf(fp, "(%0.0f, %0.0f)  ", corners[i].x, corners[i].y);
		midpoint.x += corners[i].x;
		midpoint.y += corners[i].y;
		double t = pointPolygonTest(contours[contour_num], Point2f(corners[i].x, corners[i].y), true);
		if (t >= 0)
			point2contour[i] = 0;
		else
			point2contour[i] = -t;
		if (min_num == -1)
			min_num = 0;
		else
			if (point2contour[i] < point2contour[min_num])
				min_num = i;
		t = pointPolygonTest(frame[0], Point2f(corners[i].x, corners[i].y), true);
		if (t >= 0)
			tail2contour.push_back(0);
		else
			tail2contour.push_back(-t);
	}
	vector<double> tail2contour_copy(tail2contour);
	sort(tail2contour.begin(), tail2contour.end());
	int tail = -1;
	for (int i = tail2contour.size()-1; i >= 0; i--)
	{
		if (tail2contour_copy[i] == tail2contour[0] ||
			tail2contour_copy[i] == tail2contour[1])
		{
			if (i != min_num)
			{
				tail = i;
				break;
			}
		}
	}
	//printf("min distance: %0.1lf\n", point2contour[min_num]);
	if (min_num > -1 && tail > -1)
	{
		fprintf(fp, "(%0.0lf, %0.0lf)\n", corners[min_num].x, corners[min_num].y);
		//���������ɫ���Ƴ��ǵ�
		circle(copy, corners[min_num], r, Scalar(255, 255, 255), -1, 8, 0);
		circle(copy, corners[tail], r, Scalar(0, 0, 255), -1, 8, 0);
	}
	//fprintf(fp, "midpoint: (%0.0lf, %0.0lf)  ", midpoint.x, midpoint.y);
	//fputc('\n', fp);
	fclose(fp);
	//for (int i = 1; i < big_contours.size(); i++)
	//	drawContours(mask, big_contours, i, Scalar(0, 0, 0));
	//imshow("mask", mask);

	//midpoint.x /= corners.size();
	//midpoint.y /= corners.size();
	//double dx = midpoint.x - lastpoint.x;
	//double dy = midpoint.y - lastpoint.y;
	//double dis = sqrt(dx*dx + dy*dy);
	//if (framenum > 1)
	//	dist += dis;
	//double speed = dist / (framenum*frametime);
	//double instspeed = dis / frametime;
	//if (framenum == 1)
	//	speed = -1;
	//lastpoint.x = midpoint.x;
	//lastpoint.y = midpoint.y;
	//cout << "average speed:" << speed << "pixels/sec, instant speed:" << instspeed << "pixels/sec" << endl;
	//��6����ʾ�����£�����
	imshow( WINDOW_NAME, copy );
	//imshow(WINDOW_NAME, g_grayImage);
}


//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText( )
{
	//�����ӭ��Ϣ��OpenCV�汾
	printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
	printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�87������ʾ������\n");
	printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
	//���һЩ������Ϣ
	printf("\n\n\n\t��ӭ������Shi-Tomasi�ǵ��⡿ʾ������\n"); 
	printf("\n\t������������۲�ͼ��Ч��\n\n");

}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(  )
{
	//��0���ı�console������ɫ
	//system("color 2F"); 
	time_t st, ed;
	st = time(NULL);

	//��0����ʾ��������
	ShowHelpText();
	
	FILE *fp;
	fp = fopen("result.txt", "w");
	fclose(fp);
	VideoCapture capture("1.1.avi");
	if (capture.isOpened())	// ����ͷ��ȡ�ļ�����
	{
		while (true)
		{
			//��1������Դͼ�񲢽���ת��Ϊ�Ҷ�ͼ
			capture >> g_srcImage;

			if (g_srcImage.empty())
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			framenum++;          
			//g_srcImage = imread("2.jpg", 1);
			//���Ը�ʴ����
			Mat ele = getStructuringElement(MORPH_RECT, Size(3, 3));
			erode(g_srcImage, g_srcImage, ele);

			cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);

			//��ֵ�˲���ȥ���޹ص�����
			//GaussianBlur(g_grayImage, g_grayImage, Size(13, 9), 0, 0);
			medianBlur(g_grayImage, g_grayImage, 9);
			
			if (framenum == 1)
			{
				last2 = g_grayImage.clone();
				continue;
			}
			if (framenum == 2)
			{
				last1 = g_grayImage.clone();
				continue;
			}

			//��ֵ��
			threshold(g_grayImage, g_binImage, thres, 255, THRESH_BINARY);
			//Ѱ������
			vector<Vec4i> hierarchy;
			findContours(g_binImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			Mat draw = Mat::zeros(g_binImage.size(), CV_8UC3);
			//cout << "Area: ";
			double maxArea = 0;
			contour_num = 0;
			for (int i = 1; i < contours.size(); i++)
			{
				drawContours(draw, contours, i, Scalar(255, 0, 0));
				double area = contourArea(contours[i]);
					if (area > maxArea)
					{
						contour_num = i;
						maxArea = area;
					}
				//cout << area<<" ";
			}
			//cout << endl;

			//if (contours.size() > 2)
			//{
			//	cout << "contour num = " << contour_num << endl;
			//	system("pause");
			//}


			//��2���������ںͻ���������������ʾ�ͻص�������ʼ��
			if (maxArea >= 50)
			{
				namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
				createTrackbar("���ǵ���", WINDOW_NAME, &g_maxCornerNumber, g_maxTrackbarNumber, on_GoodFeaturesToTrack);
				//imshow(WINDOW_NAME, g_srcImage);
				on_GoodFeaturesToTrack(0, 0);

			}
			
			last2 = last1.clone();
			last1 = g_grayImage.clone();
			last2_color = last1_color.clone();
			last1_color = g_srcImage.clone();
			imshow("draw", draw);
			int c = waitKey(20);
			if ((char)c == 27)
			{
				break;
			}
		}
	}
	//waitKey(0);
	ed = time(NULL);
	printf("Running %ld\n", (ed - st));
	return(0);
}

void frame_diff(Mat gray1, Mat gray2, Mat gray3)
{
	Mat img_dst;
	Mat gray_diff1, gray_diff2;//�洢2�������ͼƬ
	Mat gray;//������ʾǰ����
	Mat img;//��ʾ͹��

	vector<Point> points;
	bool pause = false;
	subtract(gray2, gray1, gray_diff1);//�ڶ�֡����һ֡
	subtract(gray3, gray2, gray_diff2);//����֡���ڶ�֡

									   /*
									   gray_diff1��gray_diff2�ֱ�Ϊһ�������ĵ�֡��
									   ����Ҫ��������ֵ�Ĵ�С��ϵ���ж�ֵ������
									   ������ֵ����ʾΪ�ɼ��ı仯�������Ǵ�����ͼ�����ð�ɫ��ʾ��
									   С����ֵ��Ϊ������Ϊû�б仯���ú�ɫ��ʾ
									   */

	points.clear();

	for (int i = 0; i<gray_diff1.rows; i++)
		for (int j = 0; j<gray_diff1.cols; j++)
		{
			if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//
				gray_diff1.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
			else gray_diff1.at<unsigned char>(i, j) = 0;

			if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
				gray_diff2.at<unsigned char>(i, j) = 255;
			else gray_diff2.at<unsigned char>(i, j) = 0;
		}
	/*�������õ�ͼ���ʾ��������֡�ж������仯�Ĳ��֣��룩*/
	bitwise_and(gray_diff1, gray_diff2, gray);

	/*��ʾ֡����*/
	imshow("foreground", gray);

	/*Ϊ���ں������㣬���֡����ҵ��Ĳ��ֵ�͹��*/

	/*�����ɫ���ֵ����е�*/
	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			Point pt;
			if (gray.at<unsigned char>(i, j) == 255)
			{
				pt.x = j;
				pt.y = i;
				points.push_back(pt);
			}
		}
	}

	vertex = Mat::zeros(gray.size(), CV_8UC1);

	//����͹�� 
	vector<int> hull;
	convexHull(Mat(points), hull, true);
	int count = (int)hull.size();//͹���е�ĸ���
	if (count == 0)
		return;
	Point pt0 = points[hull[count - 1]];
	img = gray1.clone();
	//��͹��
	Point* pt = new Point[count];
	for (int i = 0; i < count; i++)
	{
		pt[i] = points[hull[i]];
	}
	const Point* ppt[1] = { pt };
	int npt[] = { count };
	fillPoly(vertex, ppt, npt, 1, Scalar(255, 255, 255));
	delete[] pt;
}