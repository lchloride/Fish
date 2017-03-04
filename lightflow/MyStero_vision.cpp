// MyCalibration2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/video/video.hpp>
//#include <opencv/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv/core/core.hpp>
#include <opencv/cvaux.h>	
#include<iostream>
using namespace std;
using namespace cv;

#undef _GLIBCXX_DEBUG

#include "cxmisc.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>

static void StereoCalib(const char* imageList, int useUncalibrated);
CvMat* mx1;
CvMat* my1;
CvMat* mx2;
CvMat* my2;

CvMat matQ;
bool isVerticalStereo = false;
//CvMat* pair;
CvSize imageSize = {0,0};

//cvMat part;
////void saveXYZ(const char* filename, const Mat& mat)
////{
////	const double max_z = 1.0e4;
////	FILE* fp = fopen(filename, "wt");
////	fprintf(fp, "\"X\"	\"Y\"	\"Z\"\n");
////	for(int y = 0; y < mat.rows; y++)
////	{
////		for(int x = 0; x < mat.cols; x++)
////		{
////			Vec3f point = mat.at<Vec3f>(y, x);
////			if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
////			fprintf(fp, "%f %f %f\n", point[0]*1000, point[1]*1000, point[2]*1000);
////		}
////	}
////	fclose(fp);
////}

int main(int argc,char** argv)
{
	const char* imageList = "E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\Calibration.txt";
	StereoCalib(imageList,0);

	printf("��ʾ���ͼ!==============");
	//IplImage* srcLeft = cvLoadImage("E:\\Trash\\L.jpg",1); 
	//IplImage* srcRight = cvLoadImage("E:\\Trash\\R.jpg",1); 
	const char* img1_filename = "E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\left01.jpg";
	const char* img2_filename = "E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\right01.jpg";

	//const char* disparity_filename = "E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\disparity_filename.yml";
	//const char* point_cloud_filename = "E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\point_cloud_filename.yml";

//=====================������ʾУ��ͼƬ�Ĵ���===============================
	/*cvNamedWindow("rectified!");
	if (!isVerticalStereo)
		pair = cvCreateMat(imageSize.height, imageSize.width*2, CV_8UC3);
	else
		pair = cvCreateMat(imageSize.height*2, imageSize.width, CV_8UC3);*/


	enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2 };
	int alg = STEREO_SGBM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	bool no_display = false;

	StereoBM bm;
	StereoSGBM sgbm;

	int color_mode = alg == STEREO_BM ? 0 : -1;
	Mat img1 = imread(img1_filename, color_mode);
	Mat img2 = imread(img2_filename, color_mode);
	Size img_size = img1.size();

	Rect roi1, roi2;
	Mat Q;

	Mat img1r, img2r;
	remap(img1, img1r, mx1, my1, INTER_LINEAR);
	remap(img2, img2r, mx2, my2, INTER_LINEAR);

	img1 = img1r;
	img2 = img2r;

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : img_size.width/8;

	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	int cn = img1.channels();

	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = bm.state->speckleWindowSize;
	sgbm.speckleRange = bm.state->speckleRange;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = alg == STEREO_HH;

	Mat disp, disp8;

	int64 t = getTickCount();
	if( alg == STEREO_BM )
		bm(img1, img2, disp);
	else
		sgbm(img1, img2, disp);
	t = getTickCount() - t;
	printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	if( !no_display )
	{
		namedWindow("left", 1);
		imshow("left", img1);
		namedWindow("right", 1);
		imshow("right", img2);
		namedWindow("disparity", 0);
		imshow("disparity", disp8);
		printf("press any key to continue...");
		fflush(stdout);
		waitKey();
		printf("\n");
	}

	/*
	if (!isVerticalStereo)
	{
		cvGetCols(pair,&part,0,imageSize.width);
		cvCvtColor(img1,&part,CV_GRAY2BGR);
		cvGetCols(pair,&part,imageSize.width,imageSize.width*2);
		cvCvtColor(img2,&part,CV_GRAY2BGR);
		for (int i=0;i<imageSize.height;i+=16)
		{
			cvLine(pair,cvPoint(0,i));
			cvPoint(imageSize.width*2,i);
			CV_RGB(0,255,0);
		}
	}
	else
	{
		cvGetRows(pair,&part,0,imageSize.height);
		cvCvtColor(img1,&part,CV_GRAY2BGR);
		cvGetRows(pair,&part,imageSize.height,imageSize.height*2);
		cvCvtColor(img2,&part,CV_GRAY2BGR);
		for (int j=0;j<imageSize.height;j+=16)
		{
			cvLine(pair,cvPoint(0,j));
			cvPoint(j,imageSize.height*2);
			CV_RGB(0,255,0);
		}
	}
	cvShowImage("rectified",pair);
	waitKey();*/

	//cvReleaseImage(&img1);
	//cvReleaseImage(&img2);
	//cvReleaseMat(&mx1);
	//cvReleaseMat(&my1);
	//cvReleaseMat(&mx2);
	//cvReleaseMat(&my2);
	//cvReleaseImage(&img1r);
	//cvReleaseImage(&img2r);


	//if(disparity_filename)
	//	imwrite(disparity_filename, disp8);

	////if(point_cloud_filename)
	////{
	////	printf("storing the point cloud...");
	////	fflush(stdout);
	////	Mat xyz;//xyz=_3dImage
	////	reprojectImageTo3D(disp8, xyz, &matQ, true);
	////	saveXYZ(point_cloud_filename, xyz);
	////	waitKey(0);
	////	printf("\n");
	////}




	//CvMat* mat_left;mat_right,mat_l,mat_r;
	//mat_left = cvCreateMat(srcLeft->height,srcLeft->width,CV_32SC1);
	//mat_right = cvCreateMat(srcRight->height,srcRight->width,CV_32SC1);
	//cvConvert(srcLeft,mat_left);
	//cvConvert(srcRight,mat_right);
	/************************************************************************/
	/* void cvRemap( const CvArr* src, CvArr* dst,
	const CvArr* mapx, const CvArr* mapy,
	int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,
	CvScalar fillval=cvScalarAll(0) );                                                                     */
	/************************************************************************/

	//mat_left = mat_l;
	//mat_right = mat_r;

	////IplImage* leftImage = cvCreateImage(cvGetSize(srcLeft), IPL_DEPTH_8U, 1);
	////IplImage* rightImage = cvCreateImage(cvGetSize(srcRight), IPL_DEPTH_8U, 1);

	////IplImage* left = NULL;// = cvCreateImage(cvGetSize(srcLeft), IPL_DEPTH_8U, 1);
	////IplImage* right = NULL;// = cvCreateImage(cvGetSize(srcRight), IPL_DEPTH_8U, 1);

	//////cvGetImage(mat_left,leftImage);
	//////cvGetImage(mat_right,rightImage);

	////cvCvtColor(srcLeft,leftImage,CV_BGR2GRAY);
	////cvCvtColor(srcRight,rightImage,CV_BGR2GRAY);

	////cvRemap(leftImage,left,mx1,my1,INTER_LINEAR);
	////cvRemap(rightImage,right,mx2,my2,INTER_LINEAR);

	////CvSize size = cvGetSize(srcLeft);
	////CvMat* disparity_left = cvCreateMat( size.height, size.width, CV_16S );
	////CvMat* disparity_right = cvCreateMat( size.height, size.width, CV_16S );

	////CvStereoGCState* state = cvCreateStereoGCState( 16, 20 );
	////cvFindStereoCorrespondenceGC( left, right,disparity_left, disparity_right, state, 0 );
	////cvReleaseStereoGCState( &state );

	////CvMat* disparity_left_visual = cvCreateMat( size.height, size.width, CV_8U );
	////cvConvertScale( disparity_left, disparity_left_visual, -16);
	////cvSaveImage("disparity.jpg", disparity_left_visual);

	////cvNamedWindow("LeftImage",1); 
	////cvNamedWindow("RightImage",1); 
	////cvNamedWindow("disparityImage",1); 

	////while(1)
	////{
	////	cvShowImage("LeftImage",left); 
	////	cvShowImage("RightImage",right); 
	////	cvShowImage("disparityImage",disparity_left_visual);
	////	if(cvWaitKey(20)==27) 
	////		break; 
	////}

	return 0;
}

static void StereoCalib(const char* imageList, int useUncalibrated)
{
	CvRect roi1, roi2;
	int nx = 0, ny = 0;
	int displayCorners = 1;
	int showUndistorted = 1;
	//OpenCV can handle left-right
	//or up-down camera arrangements
	const int maxScale = 1;
	const float squareSize = 1.f; //Set this to your actual square size
	FILE* f = fopen(imageList, "rt");
	int i, j, lr, nframes = 0, n, N = 0;
	vector<string> imageNames[2];//����ļ�������������
	vector<CvPoint3D32f> objectPoints;//����������������
	vector<CvPoint2D32f> points[2];//
	vector<CvPoint2D32f> temp_points[2];//
	vector<int> npoints;//��ŵ���������
	//    vector<uchar> active[2];
	int is_found[2] = {0, 0};//����Ϊ2����������Ԫ�طֱ��Ŵ���������ͼ���ҵ��Ľǵ����
	vector<CvPoint2D32f> temp;
	
	// ARRAY AND VECTOR STORAGE:
	double M1[3][3], M2[3][3], D1[5], D2[5];//�ڲξ���
	double R[3][3], T[3], E[3][3], F[3][3];//��ξ���
	double Q[4][4];
	CvMat _M1 = cvMat(3, 3, CV_64F, M1 );
	CvMat _M2 = cvMat(3, 3, CV_64F, M2 );
	CvMat _D1 = cvMat(1, 5, CV_64F, D1 );
	CvMat _D2 = cvMat(1, 5, CV_64F, D2 );//������任Ϊ����
	CvMat matR = cvMat(3, 3, CV_64F, R );
	CvMat matT = cvMat(3, 1, CV_64F, T );
	CvMat matE = cvMat(3, 3, CV_64F, E );
	CvMat matF = cvMat(3, 3, CV_64F, F );
	matQ = cvMat(4, 4, CV_64FC1, Q);

	char buf[1024];

	if( displayCorners )
		cvNamedWindow( "corners", 1 );
	// READ IN THE LIST OF CHESSBOARDS:
	if( !f )
	{
		fprintf(stderr, "can not open file %s\n", imageList );
		return;
	}

	//char *fgets----Get a string from a stream.���ļ��ж�ȡͼ���������buf��
	//int sscanf----Read formatted data from a string,��ȡͼ���еĽǵ����,�����nx,ny��
	if( !fgets(buf, sizeof(buf)-3, f) || sscanf(buf, "%d%d", &nx, &ny) != 2 )
		return;
	n = nx*ny;
	//resizeΪ�������·����С
	temp.resize(n);
	temp_points[0].resize(n);//��-�ܽǵ���
	temp_points[1].resize(n);//��-�ܽǵ���

	//�ֱ��ȡ��������ͼ���ͼ����
	for(i=0;;i++)
	{
		int count = 0, result=0;
		lr = i % 2;
		vector<CvPoint2D32f>& pts = temp_points[lr];//points[lr];lr=0---��.lr=1---��
		//fgets��ȡֱ��(����,�������, or until the number of characters read is equal to n �C 1)��������һ���ͽ���
		if( !fgets( buf, sizeof(buf)-3, f ))
			break;
		size_t len = strlen(buf);
		while( len > 0 && isspace(buf[len-1]))//���len>0��buf���һ��Ԫ���ǿո�
			buf[--len] = '\0';
		if( buf[0] == '#')
			continue;
		IplImage* img = cvLoadImage( buf, 0 );
		if( !img )
			break;
		imageSize = cvGetSize(img);
		imageNames[lr].push_back(buf);
		//////////////////////////////////////////////////////////////////////////
		//imageNames[0]���汣����ͼ��
		//imageNames[1]���汣����ͼ��
		//FIND CHESSBOARDS AND CORNERS THEREIN:
		for( int s = 1; s <= maxScale; s++ )
		{
			IplImage* timg = img;
			if( s > 1 )
			{
				timg = cvCreateImage(cvSize(img->width*s,img->height*s),
					img->depth, img->nChannels );
				cvResize( img, timg, CV_INTER_CUBIC );
			}
			result = cvFindChessboardCorners( timg, cvSize(nx, ny),
				&temp[0], &count,
				CV_CALIB_CB_ADAPTIVE_THRESH |
				CV_CALIB_CB_NORMALIZE_IMAGE);
			if( timg != img )
				cvReleaseImage( &timg );
			if( result || s == maxScale )
				for( j = 0; j < count; j++ )
				{
					temp[j].x /= s;
					temp[j].y /= s;
				}
				if( result )
					break;
		}
		if( displayCorners )
		{
			printf("%s\n", buf);
			IplImage* cimg = cvCreateImage( imageSize, 8, 3 );
			cvCvtColor( img, cimg, CV_GRAY2BGR );
			cvDrawChessboardCorners( cimg, cvSize(nx, ny), &temp[0],
				count, result );
			IplImage* cimg1 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
			cvResize(cimg, cimg1);
			cvShowImage( "corners", cimg1 );
			cvReleaseImage( &cimg );
			cvReleaseImage( &cimg1 );
			int c = cvWaitKey(1000);
			if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
				exit(-1);
		}
		else
			putchar('.');
		//N = pts.size();
		//pts.resize(N + n, cvPoint2D32f(0,0));
		//active[lr].push_back((uchar)result);
		is_found[lr] = result > 0 ? 1 : 0;
		//assert( result != 0 );
		if( result )
		{
			//Calibration will suffer without subpixel interpolation
			cvFindCornerSubPix( img, &temp[0], count,
				cvSize(11, 11), cvSize(-1,-1),
				cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
				30, 0.01) );
			copy( temp.begin(), temp.end(), pts.begin() );
		}
		cvReleaseImage( &img );
		//ÿ������ͼ����нǵ���ȡ֮��,�����������ͼ����ƥ��
		if(lr)
		{
			//�ж�����ͼ�Ľǵ��Ƿ����ҵ�,�����ҵ��������,�������
			if(is_found[0] == 1 && is_found[1] == 1)
			{
				assert(temp_points[0].size() == temp_points[1].size());//�ж�������ͼ�еĽǵ����Ƿ����
				int current_size = points[0].size();

				points[0].resize(current_size + temp_points[0].size(), cvPoint2D32f(0.0, 0.0));
				points[1].resize(current_size + temp_points[1].size(), cvPoint2D32f(0.0, 0.0));

				copy(temp_points[0].begin(), temp_points[0].end(), points[0].begin() + current_size);
				copy(temp_points[1].begin(), temp_points[1].end(), points[1].begin() + current_size);

				nframes++;

				printf("Pair successfully detected...\n");//������
			}

			is_found[0] = 0;
			is_found[1] = 0;

		}
	}
	fclose(f);
	printf("\n");
	printf("----------------------------nframes==%d",nframes);
	// HARVEST CHESSBOARD 3D OBJECT POINT LIST:
	objectPoints.resize(nframes*n);
	//��ʼ�����̵���������
	for( i = 0; i < ny; i++ )
		for( j = 0; j < nx; j++ )
			objectPoints[i*nx + j] =
			cvPoint3D32f(i*squareSize, j*squareSize, 0);
	for( i = 1; i < nframes; i++ )
		copy( objectPoints.begin(), objectPoints.begin() + n,
		objectPoints.begin() + i*n );
	npoints.resize(nframes,n);
	N = nframes*n;
	CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
	CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
	CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
	CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
	cvSetIdentity(&_M1);
	cvSetIdentity(&_M2);
	cvZero(&_D1);
	cvZero(&_D2);

	// CALIBRATE THE STEREO CAMERAS
	printf("Running stereo calibration ...");
	fflush(stdout);
	/* Computes the transformation from one camera coordinate system to another one
   from a few correspondent views of the same calibration target. Optionally, calibrates
   both cameras */
	cvStereoCalibrate( &_objectPoints, &_imagePoints1,
		&_imagePoints2, &_npoints,
		&_M1, &_D1, &_M2, &_D2,
		imageSize, &matR, &matT, &matE, &matF,
		cvTermCriteria(CV_TERMCRIT_ITER+
		CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_FIX_ASPECT_RATIO +
		CV_CALIB_ZERO_TANGENT_DIST +
		CV_CALIB_SAME_FOCAL_LENGTH +
		CV_CALIB_FIX_K3);
	printf(" done\n");

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	vector<CvPoint3D32f> lines[2];
	points[0].resize(N);
	points[1].resize(N);
	_imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
	_imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
	lines[0].resize(N);
	lines[1].resize(N);
	CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
	CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);
	//Always work in undistorted space
	/* Computes the original (undistorted) feature coordinates
   from the observed (distorted) coordinates */
	cvUndistortPoints( &_imagePoints1, &_imagePoints1,
		&_M1, &_D1, 0, &_M1 );
	cvUndistortPoints( &_imagePoints2, &_imagePoints2,
		&_M2, &_D2, 0, &_M2 );
	/* For each input point on one of images
   computes parameters of the corresponding
   epipolar line on the other image */
	//cvComputeCorrespondEpilines---�����Ӧ��ļ��ߵĲ���
	cvComputeCorrespondEpilines( &_imagePoints1, 1, &matF, &_L1 );
	cvComputeCorrespondEpilines( &_imagePoints2, 2, &matF, &_L2 );
	double avgErr = 0;
	for( i = 0; i < N; i++ )
	{
		double err = fabs(points[0][i].x*lines[1][i].x +
			points[0][i].y*lines[1][i].y + lines[1][i].z)
			+ fabs(points[1][i].x*lines[0][i].x +
			points[1][i].y*lines[0][i].y + lines[0][i].z);
		avgErr += err;
	}
	printf( "avg err = %g\n", avgErr/(nframes*n) );

	// save intrinsic parameters
	CvFileStorage* fstorage = cvOpenFileStorage("E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\intrinsics.yml",
		NULL, CV_STORAGE_WRITE);
	cvWrite(fstorage, "M1", &_M1);
	cvWrite(fstorage, "D1", &_D1);
	cvWrite(fstorage, "M2", &_M2);
	cvWrite(fstorage, "D2", &_D2);
	cvReleaseFileStorage(&fstorage);

	//COMPUTE AND DISPLAY RECTIFICATION
	//У��
	if( showUndistorted )
	{
		mx1 = cvCreateMat( imageSize.height,
			imageSize.width, CV_32F );
		my1 = cvCreateMat( imageSize.height,
			imageSize.width, CV_32F );
		mx2 = cvCreateMat( imageSize.height,
			imageSize.width, CV_32F );
		my2 = cvCreateMat( imageSize.height,
			imageSize.width, CV_32F );
		CvMat* img1r = cvCreateMat( imageSize.height,
			imageSize.width, CV_8U );
		CvMat* img2r = cvCreateMat( imageSize.height,
			imageSize.width, CV_8U );
		//CvMat* disp = cvCreateMat( imageSize.height,
		//	imageSize.width, CV_16S );
		double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
		CvMat _R1 = cvMat(3, 3, CV_64F, R1);
		CvMat _R2 = cvMat(3, 3, CV_64F, R2);
		// IF BY CALIBRATED (BOUGUET'S METHOD)
		if( useUncalibrated == 0 )
		{
			CvMat _P1 = cvMat(3, 4, CV_64F, P1);
			CvMat _P2 = cvMat(3, 4, CV_64F, P2);
			/* Computes 3D rotations (+ optional shift) for each camera coordinate system to make both
			   views parallel (=> to make all the epipolar lines horizontal or vertical) 
			   ˫ĿУ���Ǹ�������ͷ������õĵ�Ŀ�ڲ����ݣ����ࡢ����ԭ�㡢����ϵ������˫Ŀ���λ�ù�ϵ
			   ����ת�����ƽ�����������ֱ��������ͼ��������������ж�׼��ʹ��������ͼ�ĳ���ԭ������һ��
			   ��CV_CALIB_ZERO_DISPARITY ��־λ����ʱ�������ã���������ͷ����ƽ�С����ҳ���ƽ�湲�桢
			   �Լ����ж��� 
			   */

			//У����õ��ı任����Q��Q[0][3]��Q[1][3]�洢����У����������ͷ��ԭ�����꣨principal point��cx��cy��
			//Q[2][3]�ǽ���f��
			cvStereoRectify( &_M1, &_M2, &_D1, &_D2, imageSize,
				&matR, &matT,
				&_R1, &_R2, &_P1, &_P2, &matQ,
				CV_CALIB_ZERO_DISPARITY,
				1, imageSize, &roi1, &roi2);

			CvFileStorage* file = cvOpenFileStorage("E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\extrinsics.yml",
				NULL, CV_STORAGE_WRITE);
			cvWrite(file, "R", &matR);
			cvWrite(file, "T", &matT);    
			cvWrite(file, "R1", &_R1);
			cvWrite(file, "R2", &_R2);
			cvWrite(file, "P1", &_P1);    
			cvWrite(file, "P2", &_P2);    
			cvWrite(file, "Q", &matQ);
			cvReleaseFileStorage(&file);

			isVerticalStereo = fabs(P2[1][3]) > fabs(P2[0][3]);
			if(!isVerticalStereo)
				roi2.x += imageSize.width;
			else
				roi2.y += imageSize.height;
			//Precompute maps for cvRemap()
			/* Computes undistortion+rectification map for a head of stereo camera */
			cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_P1,mx1,my1);
			cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_P2,mx2,my2);

			////CvFileStorage* file1 = cvOpenFileStorage("E:\\cOpenCVPro\\hellochess\\MyStereo_vision\\res\\mx1.yml",
			////	NULL, CV_STORAGE_WRITE);
			////cvWrite(file1, "mx1", mx1);
		}
		//OR ELSE HARTLEY'S METHOD
		else
			assert(0);

		cvReleaseMat( &img1r );
		cvReleaseMat( &img2r );
		/*cvReleaseMat( &disp );*/
	}
}


