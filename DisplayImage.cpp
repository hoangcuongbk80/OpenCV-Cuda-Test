#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

using cv::cuda::GpuMat;
using namespace cv;
//using namespace cv::cuda;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image, imageGray, out;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::cvtColor(image, imageGray, CV_BGR2GRAY);
    cv::imshow("Input Img", image);
    cv::imshow("Input Img2", imageGray);
    cv::waitKey(0);
    std::cerr << "img type: " << imageGray.type() << std::endl;

    //Ptr<CannyEdgeDetector> detector = cv::cuda::createCannyEdgeDetector(10, 100);

    GpuMat gpuImageGray(imageGray), gpuOut, bilateralOut;
    //detector->detect(gpuImageGray, gpuOut);

    cv::cuda::bilateralFilter(gpuImageGray, bilateralOut, 7, 150, 150);
    cv::Mat reconvert(bilateralOut);
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("out", reconvert);

    waitKey(0);

    return 0;
}

/*
#include <iostream>

#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

int main()
{
    cout << "This program demonstrates using alphaComp" << endl;
    cout << "Press SPACE to change compositing operation" << endl;
    cout << "Press ESC to exit" << endl;

    namedWindow("First Image", WINDOW_NORMAL);
    namedWindow("Second Image", WINDOW_NORMAL);
    namedWindow("Result", WINDOW_OPENGL);

    setGlDevice();

    Mat src1(640, 480, CV_8UC4, Scalar::all(0));
    Mat src2(640, 480, CV_8UC4, Scalar::all(0));

    rectangle(src1, Rect(50, 50, 200, 200), Scalar(0, 0, 255, 128), 30);
    rectangle(src2, Rect(100, 100, 200, 200), Scalar(255, 0, 0, 128), 30);

    GpuMat d_src1(src1);
    GpuMat d_src2(src2);

    GpuMat d_res;

    imshow("First Image", src1);
    imshow("Second Image", src2);

    int alpha_op = ALPHA_OVER;

    const char* op_names[] =
    {
        "ALPHA_OVER", "ALPHA_IN", "ALPHA_OUT", "ALPHA_ATOP", "ALPHA_XOR", "ALPHA_PLUS", "ALPHA_OVER_PREMUL", "ALPHA_IN_PREMUL", "ALPHA_OUT_PREMUL",
        "ALPHA_ATOP_PREMUL", "ALPHA_XOR_PREMUL", "ALPHA_PLUS_PREMUL", "ALPHA_PREMUL"
    };

    for(;;)
    {
        cout << op_names[alpha_op] << endl;

        alphaComp(d_src1, d_src2, d_res, alpha_op);

        imshow("Result", d_res);

        char key = static_cast<char>(waitKey());

        if (key == 27)
            break;

        if (key == 32)
        {
            ++alpha_op;

            if (alpha_op > ALPHA_PREMUL)
                alpha_op = ALPHA_OVER;
        }
    }

    return 0;
}*/
