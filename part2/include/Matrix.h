#pragma once

#ifndef Matrix_H
#define Matrix_H

#include <vector>
#include <cstddef>

//基于int行列的矩阵无法生成矩阵，矩阵索引超出范围，修改为size_t类型，还是无法正常生成，排除int的问题
//将矩阵索引的运算符重载内的cerr改为throw,还是无法解决问题
//啊，原来是默认构造函数里的行和列写反了，改了还是不行
//原来print()函数里写成了两个++i，行了

class Matrix
{
public:
    std::vector<float> elements;
    size_t rows;
    size_t colums;
    
    Matrix(){rows = 0; colums = 0;};
    Matrix(size_t r,size_t c);//默认构造函数
    Matrix(const std::vector<std::vector<float>> mat);
    ~Matrix();
    size_t get_rows() const{return rows;};
    size_t get_colums() const{return colums;};
    float& operator()(size_t r,size_t c);
    const float& operator()(size_t r,size_t c) const;
    Matrix operator+(const Matrix& other);//矩阵加法,重载运算符"+"
    Matrix operator*(const Matrix& other);//矩阵乘法,重载运算符"*"
    void print() const;//打印矩阵
    friend Matrix RELU(Matrix target);//RELU函数
    friend Matrix softmax(Matrix target);//softmax函数
};



#endif