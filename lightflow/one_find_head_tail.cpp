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
#include <iostream>
#include <cstdio>
using namespace cv;
using namespace std;

//-----------------------------------���궨�岿�֡�-------------------------------------------- 
//  ����������һЩ������ 
//----------------------------------------------------------------------------------------------
#define WINDOW_NAME "��Shi-Tomasi�ǵ��⡿"        //Ϊ���ڱ��ⶨ��ĺ� 
#define AREA_LIMIT 50


//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//          ������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_grayImage, g_binImage;
int g_maxCornerNumber = 33;
int g_maxTrackbarNumber = 500;
RNG g_rng(12345);//��ʼ�������������
double dist = 0;
Point2f lastpoint;
int framenum = 0;
const double frametime = 0.0666666666666667;
const int thres = 27;
vector<vector<Point> > contours;
vector<vector<Point> > big_contours;
int contour_num=1;
Mat mask;
int ds = 20;



//-----------------------------��on_GoodFeaturesToTrack( )������----------------------------
//          ��������Ӧ�������ƶ���Ϣ�Ļص�����
//----------------------------------------------------------------------------------------------
void on_GoodFeaturesToTrack( int, void* )
{
	//��1���Ա���С�ڵ���1ʱ�Ĵ���
	if( g_maxCornerNumber <= 1 ) { g_maxCornerNumber = 1; }
	g_maxCornerNumber = 2;
	//��2��Shi-Tomasi�㷨��goodFeaturesToTrack�������Ĳ���׼��
	vector<Point2f> corners;
	double qualityLevel = 0.01;//�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 50;//�ǵ�֮�����С����
	int blockSize = 7;//���㵼������ؾ���ʱָ��������Χ
	double k = 0.04;//Ȩ��ϵ��
	Mat copy = g_srcImage.clone();	//����Դͼ��һ����ʱ�����У���Ϊ����Ȥ����
	//Mat copy = g_grayImage.clone();	//����Դͼ��һ����ʱ�����У���Ϊ����Ȥ����


	//��ֵ��
	threshold(g_grayImage, g_binImage, 30, 255, THRESH_BINARY);
	//Canny(g_grayImage, g_binImage, 80, 200);
	//imshow("canny", g_binImage);
	//Ѱ������
	vector<Vec4i> hierarchy;
	findContours(g_binImage, big_contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (big_contours.size() == 0)
		return;
	Mat draw = Mat::zeros(g_binImage.size(), CV_8UC1);
	//cout << "Area: ";
	double maxArea = 0;
	int big_contour_num = 0;
	for (int i = 1; i < big_contours.size(); i++)
	{
		//drawContours(draw, big_contours, i, Scalar(255, 0, 0));
		double area = contourArea(big_contours[i]);
		if (area > maxArea)
		{
			big_contour_num = i;
			maxArea = area;
		}
	}
	if (big_contour_num > 0)
		drawContours(draw, big_contours, big_contour_num, Scalar(0, 255, 0));

	RotatedRect box = minAreaRect(big_contours[big_contour_num]);
	Point2f vertex[4];
	Rect rect = box.boundingRect();
	Point rect_vertex[] = { Point(rect.x-ds, rect.y-ds), Point(rect.x + rect.width+ds, rect.y-ds),
										Point(rect.x + rect.width+ds, rect.y + rect.height+ds), Point(rect.x-ds, rect.y + rect.height+ds) };
	for (int i = 0; i < 4; i++)
		line(copy, rect_vertex[i], rect_vertex[(i + 1) % 4], Scalar(0, 0, 0), 2, LINE_AA);
	mask = Mat::zeros(g_grayImage.size(), CV_8UC1);
	mask(rect).setTo(255);

	//��3������Shi-Tomasi�ǵ���
	goodFeaturesToTrack( g_grayImage,//����ͼ��
		corners,//��⵽�Ľǵ���������
		g_maxCornerNumber,//�ǵ���������
		qualityLevel,//�ǵ���ɽ��ܵ���С����ֵ
		minDistance,//�ǵ�֮�����С����
		mask,//����Ȥ����
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
	int min_num = -1;
	for( int i = 0; i < corners.size(); i++ )
	{ 
		//���������ɫ���Ƴ��ǵ�
		circle( copy, corners[i], r, Scalar(g_rng.uniform(0,200), g_rng.uniform(0,200),
			g_rng.uniform(0,200)), -1, 8, 0 ); 
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
	}
	//printf("min distance: %0.1lf\n", point2contour[min_num]);
	if (min_num > -1)
	{
		fprintf(fp, "(%0.0lf, %0.0lf)\n", corners[min_num].x, corners[min_num].y);
		//���������ɫ���Ƴ��ǵ�
		circle(copy, corners[min_num], r, Scalar(255, 255, 255), -1, 8, 0);
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
	system("color 2F"); 

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
			cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);

			//��ֵ�˲���ȥ���޹ص�����
			//GaussianBlur(g_grayImage, g_grayImage, Size(13, 9), 0, 0);
			medianBlur(g_grayImage, g_grayImage, 13);
			//��ֵ��
			threshold(g_grayImage, g_binImage, thres, 255, THRESH_BINARY);
			//Ѱ������
			vector<Vec4i> hierarchy;
			findContours(g_binImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			//Mat draw = Mat::zeros(g_binImage.size(), CV_8UC3);
			Mat draw = g_binImage.clone();
			cvtColor(draw, draw, COLOR_GRAY2BGR);
			cout << "Area: ";
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
				cout << area<<" ";
			}
			cout << endl;
			imshow("draw", draw);
			int cc = waitKey(10);
			if (contours.size() > 2)
			{
				cout << "contour num = " << contour_num << endl;
				//system("pause");
			}


			//��2���������ںͻ���������������ʾ�ͻص�������ʼ��
			if (maxArea >= AREA_LIMIT)
			{
				namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
				createTrackbar("���ǵ���", WINDOW_NAME, &g_maxCornerNumber, g_maxTrackbarNumber, on_GoodFeaturesToTrack);
				imshow(WINDOW_NAME, g_srcImage);
				on_GoodFeaturesToTrack(0, 0);

			}
			int c = waitKey(50);
			if ((char)c == 27)
			{
				break;
			}
		}
	}
	//waitKey(0);
	return(0);
}