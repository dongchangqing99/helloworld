/*
���ܣ�ʹ��opencv���Լ�Dlib��
���ߣ�������
�汾��V1.0
��Ȩ������������
��ʷ��¼��
2019/11/26����ʼ���
2019/11/27����ʼ����
2019/12/03,��������
�ο����ϣ�
https://blog.csdn.net/wangxing233/article/details/51549880;	  ��Ҫ�ο�
https://blog.csdn.net/zmdsjtu/article/details/52235056��
https://blog.csdn.net/weixin_41215479/article/details/85252942��
https://blog.csdn.net/zmdsjtu/article/details/53454071;
https://blog.csdn.net/zmdsjtu/article/details/52422847;
https://blog.csdn.net/zqckzqck/article/details/78979219;
https://blog.csdn.net/zqckzqck/article/details/79040443;
https://blog.csdn.net/u012792343/article/details/78427368,���JPEGͼ���������
https://blog.csdn.net/yiyuehuan/article/details/70667318��
https://blog.csdn.net/ngy321/article/details/89453581,�����ԭ����p���ܲ���rect�ڣ����߽߱糬�������Ӹ��жϣ�
https://blog.csdn.net/yubin1277408629/article/details/53561037,DLIBתOPENCV��
http://wwwbuild.net/BigDataDigest/258755.html��
https://blog.csdn.net/u011473714/article/details/89379497��

��ע˵����
��Ҫ����Dlib���OpenCV�⣬�Ƚ��鷳���ǻ���������ԵĻ���ΪOpenCV2.4.11+VS2012+Dlib18.18

*/

#include <iostream>		//�Լ���ӵ�ͷ�ļ�
#include <Windows.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#include <dlib/image_processing/frontal_face_detector.h>		//Դ�����Դ���ͷ�ļ�
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------


//������⺯��
void faceLandmarkDetection(dlib::array2d<rgb_pixel>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{     
	dlib::frontal_face_detector detector = get_frontal_face_detector();		//Dlib���е���������������

	std::vector<dlib::rectangle> dets = detector(img);		//rectangle��ʾ�����߽�򣬻�������������
	
	full_object_detection shape = sp(img, dets[0]);		//���68�������ؼ�������

	for (int i = 0; i < shape.num_parts(); ++i)			//���ؼ�������洢��landmark������
	{
		float x=shape.part(i).x();
		float y=shape.part(i).y();	
		landmark.push_back(Point2f(x,y));		
	}
}


//���߽�4�������4���е������,�Ӷ�ƴ�ӵ�ʱ���γɱ��������ν���ƴ��
void addKeypoints(std::vector<Point2f>& points,Size imgSize)
{
	points.push_back(Point2f(1,1));
	points.push_back(Point2f(1,imgSize.height-1));
	points.push_back(Point2f(imgSize.width-1,imgSize.height-1));
	points.push_back(Point2f(imgSize.width-1,1));
	points.push_back(Point2f(1,imgSize.height/2));
	points.push_back(Point2f(imgSize.width/2,imgSize.height-1));
	points.push_back(Point2f(imgSize.width-1,imgSize.height/2));
	points.push_back(Point2f(imgSize.width/2,1));
}


//�����ں�ͼ�������������
void morpKeypoints(const std::vector<Point2f>& points1,const std::vector<Point2f>& points2,std::vector<Point2f>& pointsMorph, double alpha)
{
	for (int i = 0; i < points1.size(); i++)	//����ں�ͼ�Ĺؼ�������
	{
		float x, y;
		x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
		y = (1 - alpha) * points1[i].y + alpha * points2[i].y;
		pointsMorph.push_back(Point2f(x, y));
	}
}


//����һ���ṹ�壬���ڴ洢���
struct correspondens{
	std::vector<int> index;
};


//�ںϺ����㷨����
void delaunayTriangulation(const std::vector<Point2f>& points1,const std::vector<Point2f>& points2,std::vector<Point2f>& pointsMorph,double alpha,std::vector<correspondens>& delaunayTri,Size imgSize)
{
	morpKeypoints(points1,points2,pointsMorph,alpha);	//���ú���������ںϺ�ؼ��������
	Rect rect(0, 0, imgSize.width, imgSize.height);		//��rect������Ϊ��ͼ�ߴ磬��ֹ���������㳬����ͼ��ߴ磬������
	//for(int i=0;i<pointsMorph.size();++i)			//�������
	//{
	//	cout<<"x="<<pointsMorph[i].x<<",y= "<<pointsMorph[i].y<<endl;
	//}

	int k=0;
	cv::Subdiv2D subdiv(rect);		//���������ʷֵ㼯
	for (std::vector<Point2f>::iterator it = pointsMorph.begin(); it != pointsMorph.end(); it++)	//��76���ؼ��㸳ֵ�������ʷֵ㼯��
	{
		if((*it).x >= rect.x && (*it).y >= rect.y && (*it).x < rect.x + rect.width && (*it).y < rect.y + rect.height)	//�����ֳ����ߴ緶Χ�Ĵ���㣬�򲻸�ֵ
		{
			subdiv.insert(*it);
		}
	}
	
	std::vector<Vec6f> triangleList;		//�����޷������ţ��ú���ֻ�ǻ�ÿ������ƴ�������εĶ�������
	subdiv.getTriangleList(triangleList);		//ͨ�������ʷֺ����������Ԫ�飬��һ���������������������(x,y)���
	
	for (size_t i = 0; i < triangleList.size(); ++i)	//��76����ÿ�������εĶ������긳ֵ
	{
		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back( Point2f(t[0], t[1]) );
		pt.push_back( Point2f(t[2], t[3]) );
		pt.push_back( Point2f(t[4], t[5]) );
		//cout<<"pt.size() is "<<pt.size()<<endl;
		
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))	//ͨ��������������ԭʼͼ���ں�ͼ�����ͼһһ��Ӧ����
		{
			//cout<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<t[3]<<" "<<t[4]<<" "<<t[5]<<endl;
			int count = 0;
			for (int j = 0; j < 3; ++j)
			{
				for (size_t k = 0; k < pointsMorph.size(); k++)
				{
					if (abs(pt[j].x - pointsMorph[k].x) < 1.0   &&  abs(pt[j].y - pointsMorph[k].y) < 1.0)	//��ÿ���ҵ����������������ҳ���
					{
						ind.index.push_back(k);
						count++;
					}
				}
			}
			if (count == 3)
			{
				delaunayTri.push_back(ind);		//�����Щ��Ƭ�����εĶ����������ţ�����76���ؼ����е����кţ�ÿ��Ϊһ��
		
			}
		}
	}	
}


//���ݶ�Ӧ�������������ǵ㼯�ϼ���������Ȼ�����������ͼ����з���任ͶӰ��ȥ
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> & srcTri, std::vector<Point2f> & dstTri)
{
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);	//��ԭͼ��������任����Ϊ��ͼ
}


//����������ƴ��ͼ��˵����ͼ�񲻽����Ƕ��������������ƴ�ӣ����Ƕ��������������������ƴ��
void morphTriangle(Mat &img1, Mat &img2, Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2, std::vector<Point2f> &t, double alpha)
{
	Rect r = cv::boundingRect(t);	//����������������������������������ذ��������������С������
	Rect r1 = cv::boundingRect(t1);
	Rect r2 = cv::boundingRect(t2);

	std::vector<Point2f> t1Rect, t2Rect, tRect;		//�ⲿ����Ҫ�����һ��
	std::vector<Point> tRectInt;
	for (int i = 0; i < 3; ++i)
	{
		tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
		tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y));

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);			//maskĬ��Ϊ��õ���С������ROI����
	fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);		//���ú���������͹����ν������,�������������Ϊ��ɫ

	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect);
	img2(r2).copyTo(img2Rect);
	
	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

	applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);		//���ú�������ԭʼͼ����������ƬͶӰ�����ͼ����
	applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

	Mat imgRect = (1.0 - alpha)*warpImage1 + alpha*warpImage2;		//��Ȩ������ͼ������ֵ
	
	multiply(imgRect, mask, imgRect);		//������г˷������Ŀ�ģ����ǽ������������������ֵ�����0����ֹ��ͼ����
	multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));		//�Ѿ��β����ľ��������Ƶ����ͼ����
	img(r) = img(r) + imgRect;
	
}




//ͼ���ںϳ��򣬺��ĳ���
void morp(Mat &img1, Mat &img2, Mat& imgMorph, double alpha, const std::vector<Point2f> &points1, const std::vector<Point2f> &points2, const std::vector<correspondens> &triangle)
{
	img1.convertTo(img1, CV_32F);		//ת��Ϊ������
	img2.convertTo(img2, CV_32F);

	std::vector<Point2f> points;
	morpKeypoints(points1,points2,points,alpha);		//���ú������ںϹؼ��㣬����ʵ��������������Ƭ�������ظ���
	
	int x, y, z;
	int count = 0;
	for (int i=0;i<triangle.size();++i)		//������������������Ƭ�����ںϴ���
	{
		correspondens corpd=triangle[i];
		x = corpd.index[0];		//�������εĶ��㸳ֵ
		y = corpd.index[1];
		z = corpd.index[2];
		std::vector<Point2f> t1, t2, t;
		t1.push_back(points1[x]);		//��ԭͼA���ں�ͼB�����ͼC�����ǹؼ���������ȡ����
		t1.push_back(points1[y]);
		t1.push_back(points1[z]);

		t2.push_back(points2[x]);
		t2.push_back(points2[y]);
		t2.push_back(points2[z]);

		t.push_back(points[x]);
		t.push_back(points[y]);
		t.push_back(points[z]);
		morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);	//���ú���������ͼ���ں�
	}
}


//�������������￪ʼ
int main(int argc, char** argv)
{  
 //-------------- ��һ��������ͼ�� --------------------------------------------       
	shape_predictor sp;			//����ģ�ͣ���ʼ��������λ�ؼ���ļ����sp
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	
	Mat img1o = imread("image_A.jpg");		//��ȡͼ��
	Mat img2ot = imread("image_B.jpeg");
	if(!img1o.data)
	{
		cout<<"��һ��ͼ���ȡʧ�ܣ��鿴�Ƿ�·����ȷ��"<<endl;
	}
	else
	{
		cout<<"��һ��ͼ��˳����ȡ��"<<endl;
	}
	if(!img2ot.data)
	{
		cout<<"�ڶ���ͼ���ȡʧ�ܣ��鿴�Ƿ�·����ȷ��"<<endl;
	}
	else
	{
		cout<<"�ڶ���ͼ��˳����ȡ��"<<endl;
	}
	Mat img2o;
	int imgw = img1o.rows;			//��û���ͼ�ĳߴ�
	int imgh = img1o.cols;
	dlib::array2d<rgb_pixel> img1(imgw,imgh),img2(imgw,imgh);		//����һ��array2dͼ������
	cv::resize(img2ot,img2o,cv::Size(imgh,imgw),(0,0),(0,0),1);		//���ں�ͼ�ߴ�����������ͼһ��

	for(int i=0;i<imgw;++i)			//��OpenCV��Mat����ͼ������ת����Dlib��array2d��������
	{
		for(int j=0;j<imgh;++j)
		{
			img1[i][j].blue = img1o.at<cv::Vec3b>(i,j)[0];
			img1[i][j].green = img1o.at<cv::Vec3b>(i,j)[1];
			img1[i][j].red = img1o.at<cv::Vec3b>(i,j)[2];
		}
	}
	for(int i=0;i<imgw;++i)
	{
		for(int j=0;j<imgh;++j)
		{
			img2[i][j].blue = img2o.at<cv::Vec3b>(i,j)[0];
			img2[i][j].green = img2o.at<cv::Vec3b>(i,j)[1];
			img2[i][j].red = img2o.at<cv::Vec3b>(i,j)[2];
		}
	}
	
//----------------- �ڶ���������Dlib�⣬��������ؼ��� ---------------------------------------------
	std::vector<Point2f> landmarks1,landmarks2;
	faceLandmarkDetection(img1,sp,landmarks1);		//���ú��������ͼ���⵽�����������Լ�68���ؼ���
	faceLandmarkDetection(img2,sp,landmarks2);
	
//	addKeypoints(landmarks1,img1o.size());		//����ԭͼ�߽�8����
//	addKeypoints(landmarks2,img2o.size());

	for (int i=0;i<landmarks1.size();++i)	//���Խ��ҵ��Ĺؼ��㣬��ԭͼ�л�Ȧ��ʾ����
	{
		circle(img1o, landmarks1[i], 2, CV_RGB(255, 0, 0), 1, 8, 3);
	}
	namedWindow("landmark");
	imshow("landmark",img1o);
//	waitKey();

//--------------- �������������ں� ----------------------------------------------
	std::vector<Mat> resultImage;		//���ڶ���ͼ��ʱ�������ͼ�洢�����������
	resultImage.push_back(img1o);		//�Ƚ�����ͼ�洢
	for(double alpha= 0.25;alpha<1; alpha += 0.25)		//�����ǰ���0.25�����ӵı��������ں�
	{
		Mat imgMorph = Mat::zeros(img1o.size(), CV_32FC3);		//���岢��ʼ���ں�ͼ�����
		std::vector<Point2f> pointsMorph;

		std::vector<correspondens> delaunayTri;			//����һ��2D���飬���ڴ洢������Ƭ�Ķ�����������
		delaunayTriangulation(landmarks1,landmarks2,pointsMorph,alpha,delaunayTri,img1o.size());	//���ú�������ö����������Ƭ��Ķ�������Լ�����

		morp(img1o, img2o, imgMorph, alpha, landmarks1, landmarks2, delaunayTri);		//���ú�����ͼ���ں�
//		namedWindow("�����ں�ͼ��");
//		imshow("�ںϽ��ͼ��",imgMorph);		//��ʾ�ںϽ��ͼ��
//		waitKey(0);

		resultImage.push_back(imgMorph);		//���ں�ͼ���������
	}
	resultImage.push_back(img2o);		//���ں�ͼ�´洢
	


//----------- ���Ĳ����洢�ں�ͼ�� --------------------------------
	
	for (int i=0;i<resultImage.size();++i)		//ʹ��string�ַ���ʵ���ں�ͼ������
	{
		string st="result";
		char t[20];
		sprintf(t, "%d", i);
		st=st+t;
		st=st+".jpg";
		imwrite(st,resultImage[i]);
	}
	cout<<"������"<<endl;
	system("pause");
	return 0;

}




