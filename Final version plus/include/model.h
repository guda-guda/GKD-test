#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"
#include "json.hpp"
#include <string>
#include <fstream>
#include <cstddef>
#include <iostream>

using json = nlohmann::json;

template<typename T>
class Basemodel{
public:
    virtual ~Basemodel() = default;
    virtual Matrix<T> forward(Matrix<T> input) = 0;
};

template<typename T>
class model : public Basemodel<T>{
private :
Matrix<T> weight1;
Matrix<T> bias1;
Matrix<T> weight2;
Matrix<T> bias2;

public:
model( Matrix<T> w1, Matrix<T> b1, Matrix<T> w2, Matrix<T> b2){weight1=w1;weight2=w2;bias1=b1;bias2=b2;};
model(const std::string& foldername);
~model(){};
Matrix<T> forward(Matrix<T> input) override;
};

template<typename T>
Matrix<T> model<T>::forward(Matrix<T> input)
{   
    Matrix<T> result1 = Blockmultiply_threads(RELU(Blockmultiply_threads(input,weight1,32)+bias1),weight2,32)+bias2;
    Matrix<T> result = softmax(result1);
    return result;
}

//构造基类指针
template<typename T>
Basemodel<T>* createmodel(const std::string& foldername)
{
    return new model<T>(foldername);
}

//读取二进制文件
template<typename T>
Matrix<T> read_binfile(std::string file_path,size_t r,size_t l)
{
    Matrix<T> result(r,l);
    std::ifstream binfile(file_path,std::ios::binary);
    
    if(!binfile.is_open()){
        std::cerr <<"无法打开二进制文件:" <<file_path <<std::endl;
        return result;
    }

    for(size_t i = 0;i < r;++i){
        for (size_t j = 0; j < l; ++j)
        {
            T value;
            binfile.read( reinterpret_cast<char*>(&value),sizeof(T));
            result(i,j)= value;
        }
    }
    binfile.close();
    return result;
}

template<typename T>
model<T>::model(const std::string& foldername)
{
    //打开json文件
    json j;
    std::ifstream jfile(foldername+"/meta.json");
    if(!jfile.is_open()){
        throw std::runtime_error("Failed to open meta.json");
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
    weight1 = Matrix<T>(w1_rows, w1_cols);
    bias1 = Matrix<T>(b1_rows, b1_cols);
    weight2 = Matrix<T>(w2_rows, w2_cols);
    bias2 = Matrix<T>(b2_rows, b2_cols);
    
    //读取二进制文件
    weight1=read_binfile<T>(foldername+"/fc1.weight",w1_rows,w1_cols); 
    bias1=read_binfile<T>(foldername+"/fc1.bias",b1_rows,b1_cols);
    weight2=read_binfile<T>(foldername+"/fc2.weight",w2_rows,w2_cols);
    bias2=read_binfile<T>(foldername+"/fc2.bias",b2_rows,b2_cols);
    
}
#endif