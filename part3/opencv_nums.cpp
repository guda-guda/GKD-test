#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "Matrix.h"
#include "model.h"

using namespace cv;

int main()
{
    Mat num = imread("E:\\code\\GKD-test\\part3\\nums\\2.png",IMREAD_GRAYSCALE);
    if(num.empty())
    {
        std::cerr <<"无法加载图像" <<std::endl;
        return -1;
    }
    
    //调整大小
    Mat renum;
    resize(num,renum,Size(28,28),0,0,INTER_AREA);//调用resize方法调整图片矩阵大小
    
    //矩阵转化
    Matrix mtxnum(renum);
    
    std::cout << "Matrix尺寸:rows:" << mtxnum.get_rows() << ",cols:" << mtxnum.get_colums() << std::endl;


    return 0;
}