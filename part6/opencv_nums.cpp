#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstddef>
#include <string>
#include "json.hpp"
#include "Matrix.h"
#include "model.h"

using namespace cv;
using json = nlohmann::json;

//读取二进制文件
Matrix<float> read_binfile(std::string file_path,size_t r,size_t l)
{
    Matrix<float> result(r,l);
    std::ifstream binfile(file_path,std::ios::binary);
    
    if(!binfile.is_open()){
        std::cerr <<"无法打开二进制文件:" <<file_path <<std::endl;
        return result;
    }

    for(size_t i = 0;i < r;++i){
        for (size_t j = 0; j < l; ++j)
        {
            float value;
            binfile.read( reinterpret_cast<char*>(&value),sizeof(float));
            result(i,j)= value;
        }
    }
    binfile.close();
    return result;
}

model<float> construct()
{
    json j;
    Matrix<float> w1,b1,w2,b2;
    std::ifstream jfile("E:\\code\\GKD-test\\part2\\mnist-fc\\meta.json");
    if(!jfile.is_open()){
        std::cerr <<"无法打开文件meta.json" <<std::endl;
    }
    jfile >> j;
    jfile.close();

    //读取数据作为长与宽,长是列数，宽是行数
    size_t w1_rows = j["fc1.weight"][0].get<int>();
    size_t w1_cols = j["fc1.weight"][1].get<int>();
    size_t b1_rows = j["fc1.bias"][0].get<int>();
    size_t b1_cols = j["fc1.bias"][1].get<int>();
    size_t w2_rows = j["fc2.weight"][0].get<int>();
    size_t w2_cols = j["fc2.weight"][1].get<int>();
    size_t b2_rows = j["fc2.bias"][0].get<int>();
    size_t b2_cols = j["fc2.bias"][1].get<int>();

    //用读取的长宽初始化矩阵的大小
    w1 = Matrix<float>(w1_rows, w1_cols);
    b1 = Matrix<float>(b1_rows, b1_cols);
    w2 = Matrix<float>(w2_rows, w2_cols);
    b2 = Matrix<float>(b2_rows, b2_cols);

    //读取二进制文件
    w1=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc1.weight",w1_rows,w1_cols); 
    b1=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc1.bias",b1_rows,b1_cols);
    w2=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc2.weight",w2_rows,w2_cols);
    b2=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc2.bias",b2_rows,b2_cols);

    model<float> result(w1,b1,w2,b2);
    return result;
}

int main()
{
    Mat num = imread("E:\\code\\GKD-test\\part6\\nums\\2.png",IMREAD_GRAYSCALE);
    if(num.empty())
    {
        std::cerr <<"无法加载图像" <<std::endl;
        return -1;
    }
    
    //调整大小
    Mat renum;
    resize(num,renum,Size(28,28),0,0,INTER_AREA);//调用resize方法调整图片矩阵大小
    
    //矩阵转化
    Matrix<float> mtxnum(renum);
    //std::cout << "Matrix尺寸:rows:" << mtxnum.get_rows() << ",cols:" << mtxnum.get_colums() << std::endl;
    //forward方法
    model<float> test =construct();
    Matrix<float> result=test.forward(mtxnum);
    std::cout <<"The result of the standard model is:" <<std::endl;
    result.print();
    return 0;
}