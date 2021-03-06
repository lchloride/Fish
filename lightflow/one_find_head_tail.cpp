//--------------------------------------【程序说明】-------------------------------------------
//		程序说明：《OpenCV3编程入门》OpenCV3版书本配套示例程序87
//		程序描述：Shi-Tomasi角点检测示例
//		开发测试所用操作系统： Windows 7 64bit
//		开发测试所用IDE版本：Visual Studio 2010
//		开发测试所用OpenCV版本：	3.0 beta
//		2014年11月 Created by @浅墨_毛星云
//		2014年12月 Revised by @浅墨_毛星云
//------------------------------------------------------------------------------------------------



//---------------------------------【头文件、命名空间包含部分】----------------------------
//		描述：包含程序所使用的头文件和命名空间
//------------------------------------------------------------------------------------------------
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//----------------------------------------------------------------------------------------------
#define WINDOW_NAME "【Shi-Tomasi角点检测】"        //为窗口标题定义的宏 
#define AREA_LIMIT 50


//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_grayImage, g_binImage;
int g_maxCornerNumber = 33;
int g_maxTrackbarNumber = 500;
RNG g_rng(12345);//初始化随机数生成器
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



//-----------------------------【on_GoodFeaturesToTrack( )函数】----------------------------
//          描述：响应滑动条移动消息的回调函数
//----------------------------------------------------------------------------------------------
void on_GoodFeaturesToTrack( int, void* )
{
	//【1】对变量小于等于1时的处理
	if( g_maxCornerNumber <= 1 ) { g_maxCornerNumber = 1; }
	g_maxCornerNumber = 2;
	//【2】Shi-Tomasi算法（goodFeaturesToTrack函数）的参数准备
	vector<Point2f> corners;
	double qualityLevel = 0.01;//角点检测可接受的最小特征值
	double minDistance = 50;//角点之间的最小距离
	int blockSize = 7;//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;//权重系数
	Mat copy = g_srcImage.clone();	//复制源图像到一个临时变量中，作为感兴趣区域
	//Mat copy = g_grayImage.clone();	//复制源图像到一个临时变量中，作为感兴趣区域


	//阈值化
	threshold(g_grayImage, g_binImage, 30, 255, THRESH_BINARY);
	//Canny(g_grayImage, g_binImage, 80, 200);
	//imshow("canny", g_binImage);
	//寻找轮廓
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

	//【3】进行Shi-Tomasi角点检测
	goodFeaturesToTrack( g_grayImage,//输入图像
		corners,//检测到的角点的输出向量
		g_maxCornerNumber,//角点的最大数量
		qualityLevel,//角点检测可接受的最小特征值
		minDistance,//角点之间的最小距离
		mask,//感兴趣区域
		blockSize,//计算导数自相关矩阵时指定的邻域范围
		false,//不使用Harris角点检测
		k );//权重系数


	//【4】输出文字信息
	cout<<"\t>此次检测到的角点数量为："<<corners.size()<<endl;

	//【5】绘制检测到的角点
	int r = 4;
	FILE *fp;
	fp = fopen("result.txt", "a");
	Point2f midpoint;
	vector<double> point2contour;
	point2contour.assign(corners.size(), 0);
	int min_num = -1;
	for( int i = 0; i < corners.size(); i++ )
	{ 
		//以随机的颜色绘制出角点
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
		//以随机的颜色绘制出角点
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
	//【6】显示（更新）窗口
	imshow( WINDOW_NAME, copy );
	//imshow(WINDOW_NAME, g_grayImage);
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText( )
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第87个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
	//输出一些帮助信息
	printf("\n\n\n\t欢迎来到【Shi-Tomasi角点检测】示例程序\n"); 
	printf("\n\t请调整滑动条观察图像效果\n\n");

}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(  )
{
	//【0】改变console字体颜色
	system("color 2F"); 

	//【0】显示帮助文字
	ShowHelpText();
	
	FILE *fp;
	fp = fopen("result.txt", "w");
	fclose(fp);
	VideoCapture capture("1.1.avi");
	if (capture.isOpened())	// 摄像头读取文件开关
	{
		while (true)
		{
			//【1】载入源图像并将其转换为灰度图
			capture >> g_srcImage;

			if (g_srcImage.empty())
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			framenum++;          
			//g_srcImage = imread("2.jpg", 1);
			cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);

			//中值滤波，去除无关的噪声
			//GaussianBlur(g_grayImage, g_grayImage, Size(13, 9), 0, 0);
			medianBlur(g_grayImage, g_grayImage, 13);
			//阈值化
			threshold(g_grayImage, g_binImage, thres, 255, THRESH_BINARY);
			//寻找轮廓
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


			//【2】创建窗口和滑动条，并进行显示和回调函数初始化
			if (maxArea >= AREA_LIMIT)
			{
				namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
				createTrackbar("最大角点数", WINDOW_NAME, &g_maxCornerNumber, g_maxTrackbarNumber, on_GoodFeaturesToTrack);
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