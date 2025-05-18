#include "Matrix.h"
#include "model.h"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <cstddef>
#include <string>

using json = nlohmann::json;

//读取二进制文件
Matrix read_binfile(std::string file_path,size_t r,size_t l)
{
    Matrix result(r,l);
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


int main()
{
    json j;
    Matrix w1,b1,w2,b2;
    std::ifstream jfile("E:\\code\\GKD-test\\part2\\mnist-fc\\meta.json");
    if(!jfile.is_open()){
        std::cerr <<"无法打开文件meta.json" <<std::endl;
        return 1;
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
    w1 = Matrix(w1_rows, w1_cols);
    b1 = Matrix(b1_rows, b1_cols);
    w2 = Matrix(w2_rows, w2_cols);
    b2 = Matrix(b2_rows, b2_cols);

    //测试读取到的长宽是否正确
    /*std::cout <<"w1的长宽分别是:" <<std::endl;
    std::cout <<w1.get_colums() <<" " <<w1.get_rows() <<std::endl;
    std::cout <<"b1的长宽分别是:" <<std::endl;
    std::cout <<b1.get_colums() <<" " <<b1.get_rows() <<std::endl;
    std::cout <<"w2的长宽分别是:" <<std::endl;
    std::cout <<w2.get_colums() <<" " <<w2.get_rows() <<std::endl;
    std::cout <<"b2的长宽分别是:" <<std::endl;
    std::cout <<b2.get_colums() <<" " <<b2.get_rows() <<std::endl;*/
    
    //读取二进制文件
    w1=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc1.weight",w1_rows,w1_cols); 
    b1=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc1.bias",b1_rows,b1_cols);
    w2=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc2.weight",w2_rows,w2_cols);
    b2=read_binfile("E:\\code\\GKD-test\\part2\\mnist-fc\\fc2.bias",b2_rows,b2_cols);

    //打印读取到的矩阵
    //std::cout <<"w1:" <<std::endl;
    //w1.print();
    //b1.print();
    //w2.print();
    //b2.print();

    //测试model类
    model test(w1,b1,w2,b2);
    Matrix input(1, 784);
    Matrix result=test.forward(input);
    std::cout <<"The result of the standard model is:" <<std::endl;
    result.print();

    return 0;
}


