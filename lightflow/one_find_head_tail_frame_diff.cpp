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
//#include "3.h"
#include <iostream>
#include <cstdio>
#include <ctime>
using namespace cv;
using namespace std;

//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//----------------------------------------------------------------------------------------------
#define WINDOW_NAME "【Shi-Tomasi角点检测】"        //为窗口标题定义的宏 
#define threshold_diff1 8 //设置简单帧差法阈值
#define threshold_diff2 8 //设置简单帧差法阈值



//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_grayImage, g_binImage, last2, last1;
int g_maxCornerNumber = 33;
int g_maxTrackbarNumber = 500;
RNG g_rng(12345);//初始化随机数生成器
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

//-----------------------------【on_GoodFeaturesToTrack( )函数】----------------------------
//          描述：响应滑动条移动消息的回调函数
//----------------------------------------------------------------------------------------------
void on_GoodFeaturesToTrack( int, void* )
{
	//【1】对变量小于等于1时的处理
	if( g_maxCornerNumber <= 1 ) { g_maxCornerNumber = 1; }
	g_maxCornerNumber = 3;
	//【2】Shi-Tomasi算法（goodFeaturesToTrack函数）的参数准备
	vector<Point2f> corners;
	double qualityLevel = 0.01;//角点检测可接受的最小特征值
	double minDistance = 50;//角点之间的最小距离
	int blockSize = 7;//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;//权重系数
	Mat copy = last2.clone();	//复制源图像到一个临时变量中，作为感兴趣区域
	//Mat copy = g_grayImage.clone();	//复制源图像到一个临时变量中，作为感兴趣区域

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
	//【3】进行Shi-Tomasi角点检测
	goodFeaturesToTrack( last2,//输入图像
		corners,//检测到的角点的输出向量
		g_maxCornerNumber,//角点的最大数量
		qualityLevel,//角点检测可接受的最小特征值
		minDistance,//角点之间的最小距离
		Mat(),//感兴趣区域
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
	vector<double> tail2contour;
	int min_num=-1;
	for( int i = 0; i < corners.size(); i++ )
	{ 
		//以随机的颜色绘制出角点
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
		//以随机的颜色绘制出角点
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
	//system("color 2F"); 
	time_t st, ed;
	st = time(NULL);

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
			//尝试腐蚀操作
			Mat ele = getStructuringElement(MORPH_RECT, Size(3, 3));
			erode(g_srcImage, g_srcImage, ele);

			cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);

			//中值滤波，去除无关的噪声
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

			//阈值化
			threshold(g_grayImage, g_binImage, thres, 255, THRESH_BINARY);
			//寻找轮廓
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


			//【2】创建窗口和滑动条，并进行显示和回调函数初始化
			if (maxArea >= 50)
			{
				namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
				createTrackbar("最大角点数", WINDOW_NAME, &g_maxCornerNumber, g_maxTrackbarNumber, on_GoodFeaturesToTrack);
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
	Mat gray_diff1, gray_diff2;//存储2次相减的图片
	Mat gray;//用来显示前景的
	Mat img;//显示凸包

	vector<Point> points;
	bool pause = false;
	subtract(gray2, gray1, gray_diff1);//第二帧减第一帧
	subtract(gray3, gray2, gray_diff2);//第三帧减第二帧

									   /*
									   gray_diff1和gray_diff2分别为一二和三四的帧差
									   下面要根据与阈值的大小关系进行二值化处理
									   大于阈值即表示为可见的变化，在我们处理后的图形中用白色显示；
									   小于阈值认为可以认为没有变化，用黑色显示
									   */

	points.clear();

	for (int i = 0; i<gray_diff1.rows; i++)
		for (int j = 0; j<gray_diff1.cols; j++)
		{
			if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//
				gray_diff1.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
			else gray_diff1.at<unsigned char>(i, j) = 0;

			if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
				gray_diff2.at<unsigned char>(i, j) = 255;
			else gray_diff2.at<unsigned char>(i, j) = 0;
		}
	/*我们所用的图像表示了连续两帧中都发生变化的部分（与）*/
	bitwise_and(gray_diff1, gray_diff2, gray);

	/*显示帧差法结果*/
	imshow("foreground", gray);

	/*为便于后续计算，求出帧差法所找到的部分的凸包*/

	/*求出白色部分的所有点*/
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

	//计算凸包 
	vector<int> hull;
	convexHull(Mat(points), hull, true);
	int count = (int)hull.size();//凸包中点的个数
	if (count == 0)
		return;
	Point pt0 = points[hull[count - 1]];
	img = gray1.clone();
	//画凸包
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