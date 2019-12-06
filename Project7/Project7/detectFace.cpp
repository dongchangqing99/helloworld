/*
https://blog.csdn.net/u012792343/article/details/78427368,解决JPEG图像加载问题
https://blog.csdn.net/yiyuehuan/article/details/70667318，
https://blog.csdn.net/ngy321/article/details/89453581,出错的原因是p可能不在rect内（或者边界超出），加个判断：
https://blog.csdn.net/yubin1277408629/article/details/53561037,DLIB转OPENCV
*/

#include <iostream>
#include <Windows.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include<dlib/opencv/cv_image_abstract.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<vector>

using namespace cv;
using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

//人脸检测函数
void faceLandmarkDetection(dlib::array2d<rgb_pixel>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{     
	dlib::frontal_face_detector detector = get_frontal_face_detector();		//Dlib库中的人脸检测器类对象
	//dlib::pyramid_up(img);
	
	std::vector<dlib::rectangle> dets = detector(img);		//rectangle表示人脸边界框，获得人脸轮廓框架
	//cout << "Number of faces detected: " << dets.size() << endl;
	
	full_object_detection shape = sp(img, dets[0]);		//获得68个人脸关键点坐标
	//image_window win;
	//win.clear_overlay();
	//win.set_image(img);
	//win.add_overlay(render_face_detections(shape));
	for (int i = 0; i < shape.num_parts(); ++i)			//将关键点坐标存储到landmark变量中
	{
		float x=shape.part(i).x();
		float y=shape.part(i).y();	
		landmark.push_back(Point2f(x,y));		
	}


}


/*
//add eight keypoints to the keypoints set of the input image.
//the added eight keypoints are the four corners points of the image, plus four median points of the four edges of the image.  
*/

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



/*
// calculate the keypoints on the morph image.
*/

void morpKeypoints(const std::vector<Point2f>& points1,const std::vector<Point2f>& points2,std::vector<Point2f>& pointsMorph, double alpha)
{
	cout<<"points1.size="<<points1.size()<<endl;
	for (int i = 0; i < points1.size(); i++)
		{
			float x, y;
			x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
			y = (1 - alpha) * points1[i].y + alpha * points2[i].y;

			pointsMorph.push_back(Point2f(x, y));

		}
}


//定义一个结构体，用于存储序号
struct correspondens{
	std::vector<int> index;
};

//融合核心算法程序
void delaunayTriangulation(const std::vector<Point2f>& points1,const std::vector<Point2f>& points2,
			   std::vector<Point2f>& pointsMorph,double alpha,std::vector<correspondens>& delaunayTri,Size imgSize)
{
	cout<<"begin delaunayTriangulation......"<<endl;
	morpKeypoints(points1,points2,pointsMorph,alpha);
	cout<<"done morpKeypoints, pointsMorph has points "<<pointsMorph.size()<<endl;
	Rect rect(0, 0, imgSize.width, imgSize.height);
	cout<<"imgsize="<<imgSize.width<<","<<imgSize.height<<endl;
	cout<<"points.size="<<pointsMorph.size()<<endl;
	for(int i=0;i<pointsMorph.size();++i)
	{
		cout<<"x="<<pointsMorph[i].x<<",y= "<<pointsMorph[i].y<<endl;
	}
	cout<<"jieshu="<<endl;
	
	int k=0;
	cout<<"pointsMorph.size="<<pointsMorph.size()<<endl;
	cv::Subdiv2D subdiv(rect);
	for (std::vector<Point2f>::iterator it = pointsMorph.begin(); it != pointsMorph.end(); it++)
	{
		
	if((*it).x >= rect.x && (*it).y >= rect.y && (*it).x < rect.x + rect.width && (*it).y < rect.y + rect.height)
	{
		subdiv.insert(*it);
	}
		//cout<<"out="<<","<<*it<<endl;
		//cout<<"k="<<k<<endl;
		//k++;
//		subdiv.insert(*it);
	}
	cout<<"done subdiv add......"<<endl;
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	//cout<<"traingleList number is "<<triangleList.size()<<endl;
	

	
	//std::vector<Point2f> pt;
	//correspondens ind;
	for (size_t i = 0; i < triangleList.size(); ++i)
	{
		
		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back( Point2f(t[0], t[1]) );
		pt.push_back( Point2f(t[2], t[3]) );
		pt.push_back( Point2f(t[4], t[5]) );
		//cout<<"pt.size() is "<<pt.size()<<endl;
		
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			//cout<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<t[3]<<" "<<t[4]<<" "<<t[5]<<endl;
			int count = 0;
			for (int j = 0; j < 3; ++j)
				for (size_t k = 0; k < pointsMorph.size(); k++)
					if (abs(pt[j].x - pointsMorph[k].x) < 1.0   &&  abs(pt[j].y - pointsMorph[k].y) < 1.0)
					{
						ind.index.push_back(k);
						count++;
					}
			if (count == 3)
				//cout<<"index is "<<ind.index[0]<<" "<<ind.index[1]<<" "<<ind.index[2]<<endl;
				delaunayTri.push_back(ind);
		}
		//pt.resize(0);
		//cout<<"delaunayTri.size is "<<delaunayTri.size()<<endl;
	}	
	
	
}


/*
// apply affine transform on one triangle.
*/
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> & srcTri, std::vector<Point2f> & dstTri)
{
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);
}


/*
//the core function of face morph.
//morph the two input image to the morph image by transacting the set of triangles in the two input image to the morph image.
*/
void morphTriangle(Mat &img1, Mat &img2, Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2, std::vector<Point2f> &t, double alpha)
{
	Rect r = cv::boundingRect(t);
	Rect r1 = cv::boundingRect(t1);
	Rect r2 = cv::boundingRect(t2);

	std::vector<Point2f> t1Rect, t2Rect, tRect;
	std::vector<Point> tRectInt;
	for (int i = 0; i < 3; ++i)
	{
		tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
		tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y));

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
	fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect);
	img2(r2).copyTo(img2Rect);

	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

	applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
	applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

	Mat imgRect = (1.0 - alpha)*warpImage1 + alpha*warpImage2;

	multiply(imgRect, mask, imgRect);
	multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));
	img(r) = img(r) + imgRect;
}




/*
//morp the two input images into the morph image.
//first get the keypoints correspondents of the set of  triangles, then call the core function. 
*/
void morp(Mat &img1, Mat &img2, Mat& imgMorph, double alpha, const std::vector<Point2f> &points1, const std::vector<Point2f> &points2, const std::vector<correspondens> &triangle)
{
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);



	std::vector<Point2f> points;
	morpKeypoints(points1,points2,points,alpha);
	

	int x, y, z;
	int count = 0;
	for (int i=0;i<triangle.size();++i)
	{
		correspondens corpd=triangle[i];
		x = corpd.index[0];
		y = corpd.index[1];
		z = corpd.index[2];
		std::vector<Point2f> t1, t2, t;
		t1.push_back(points1[x]);
		t1.push_back(points1[y]);
		t1.push_back(points1[z]);

		t2.push_back(points2[x]);
		t2.push_back(points2[y]);
		t2.push_back(points2[z]);

		t.push_back(points[x]);
		t.push_back(points[y]);
		t.push_back(points[z]);
		morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);
		//count++;
		//string shun = "hunhe";
		//if (count % 10 == 0 || count == triangle.size() - 1 || count == triangle.size())
		//	imwrite(shun+to_string(count)+".jpg", imgMorph);
	}

}






int main(int argc, char** argv)
{  
    
        //if (argc < 3)
        //{
        //    cout << "Give some image files as arguments to this program." << endl;
        //    return 0;
        //}

        

 //-------------- step 1. load the input two images --------------------------------------------       
	shape_predictor sp;			//加载模型，初始化人脸定位关键点的检测器sp
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	

	Mat img1CV = imread("image_A.jpg");		//读取图像
	Mat img2tmp = imread("image_B.jpeg");
	Mat img2CV;
	int imgw = img1CV.rows;			//获得基本图的尺寸
	int imgh = img1CV.cols;
	dlib::array2d<rgb_pixel> img1(imgw,imgh),img2(imgw,imgh);		//定义一个array2d图像数组
	cv::resize(img2tmp,img2CV,cv::Size(imgh,imgw),(0,0),(0,0),1);		//将融合图尺寸调整到与基本图一致

	cout<<"chicun="<<img1CV.rows<<","<<img1CV.cols<<endl;
	cout<<"chicun2="<<img2CV.rows<<","<<img2CV.cols<<endl;
	cout<<"chongzuchenggong"<<endl;
	for(int i=0;i<imgw;++i)			//将OpenCV的Mat类型图像数组转换成Dlib的array2d类型数组
	{
		for(int j=0;j<imgh;++j)
		{
			img1[i][j].blue = img1CV.at<cv::Vec3b>(i,j)[0];
			img1[i][j].green = img1CV.at<cv::Vec3b>(i,j)[1];
			img1[i][j].red = img1CV.at<cv::Vec3b>(i,j)[2];
		}
	}
	cout<<"1 success"<<endl;
	for(int i=0;i<imgw;++i)
	{
		for(int j=0;j<imgh;++j)
		{
			img2[i][j].blue = img2CV.at<cv::Vec3b>(i,j)[0];
			img2[i][j].green = img2CV.at<cv::Vec3b>(i,j)[1];
			img2[i][j].red = img2CV.at<cv::Vec3b>(i,j)[2];
		}
	}
	cout<<"2 success!"<<endl;

//	dlib::load_image(img1,"imageA.bmp");		
//	dlib::load_image(img2, "imageB.bmp");
	std::vector<Point2f> landmarks1,landmarks2;
	
	//
	//array2d<rgb_pixel> testimg(img1CV.rows,img1CV.cols);
	//for(int i=0;i<img1CV.rows;++i)
	//{
	//	for(int j=0;j<img1CV.cols;++j)
	//	{
	//		testimg[i][j].blue = img1CV.at<cv::Vec3b>(i,j)[0];
	//		testimg[i][j].green = img1CV.at<cv::Vec3b>(i,j)[1];
	//		testimg[i][j].red = img1CV.at<cv::Vec3b>(i,j)[2];
	//	}
	//}


	if(!img1CV.data || !img2CV.data)
	{
		printf("No image data \n");
//        	return -1;
	}
	else
		cout<<"image readed by opencv"<<endl;

	
//----------------- step 2. detect face landmarks ---------------------------------------------
	faceLandmarkDetection(img1,sp,landmarks1);		//调用函数，获得图像检测到的人脸轮廓以及68个关键点
	faceLandmarkDetection(img2,sp,landmarks2);
	cout<<"landmark2 number is "<<landmarks2.size()<<endl;
	
	

	//add some land marks in the edges to get better performance.
//	addKeypoints(landmarks1,img1CV.size());
//	addKeypoints(landmarks2,img2CV.size());
	
	cout<<"landmark number after added is "<<landmarks1.size()<<endl;
	
	
		
	/*for (int i=0;i<landmarks1.size();++i)
	{
		circle(img1CV, landmarks1[i], 2, CV_RGB(255, 255, 255), 1, 8, 3);
	}
	imshow("landmark",img1CV);
	cv::waitKey(0);
	*/


//--------------- step 3. face morp ----------------------------------------------
	std::vector<Mat> resultImage;		//存在多张图像时，将结果图存储在这个容器中
	resultImage.push_back(img1CV);		//先将基本图存储
	cout<<"add the first image"<<endl;
	for(double alpha= 0.25;alpha<1; alpha += 0.25)		//这里是按照0.25逐渐增加的比例进行融合
	{
		Mat imgMorph = Mat::zeros(img1CV.size(), CV_32FC3);		//定义并初始化融合图像变量
		std::vector<Point2f> pointsMorph;

		std::vector<correspondens> delaunayTri;
		delaunayTriangulation(landmarks1,landmarks2,pointsMorph,alpha,delaunayTri,img1CV.size());
		cout<<"done "<<alpha <<" delaunayTriangulation..."<<delaunayTri.size()<<endl;

		morp(img1CV, img2CV, imgMorph, alpha, landmarks1, landmarks2, delaunayTri);
		cout<<"done "<<alpha<<" morph.........."<<endl;

		resultImage.push_back(imgMorph);
		cout<<"add the"<< alpha*10 +1 <<"image"<<endl;
	}
	resultImage.push_back(img2CV);
	imshow("融合结果图像",img2CV);
	cout<<"resultImage number is"<<resultImage.size()<<endl;


//----------- step 4. write into vedio --------------------------------
	
	
	
	for (int i=0;i<resultImage.size();++i)
	{
		
		//output_src<<resultImage[i];
		string st="imageA.bmp";
		char t[20];
		sprintf(t, "%d", i);
		st=st+t;
		st=st+".jpg";
		imwrite(st,resultImage[i]);
	}
//	std::vector<Mat> pic;
	
	//for (int i=0;i<resultImage.size();++i)
	//{
	//	string filename = "imageA.bmp";
	//	char t[20];
	//	sprintf(t, "%d", i);
	//	filename=filename+t;
	//	filename=filename+".jpg";
	//	pic.push_back(imread(filename));
	//}
	
	//string vedioName="imageA.bmp";
	//vedioName = vedioName+"imageB.bmp";
	//vedioName = vedioName+ ".avi";
	//VideoWriter output_src(vedioName, CV_FOURCC('M', 'J', 'P', 'G'), 5, resultImage[0].size());
	//for (int i=0;i<pic.size();++i)
	//{
	//	
	//	output_src<<pic[i];
	//}
	//cout<<"vedio wrighted....."<<endl;

	system("pause");
	return 0;

}
