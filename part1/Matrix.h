#pragma once

#ifndef Matrix_H
#define Matrix_H

#include <vector>

class Matrix
{
private:
    std::vector<float> elements;
    int rows;
    int colums;
public:
    Matrix(int l,int r);//默认构造函数
    Matrix(const std::vector<std::vector<float>> mat);
    ~Matrix();
    int get_rows() const{return rows;};
    int get_colums() const{return colums;};
    float& operator()(int r,int c);
    const float& operator()(int r,int c) const;
    Matrix operator+(const Matrix& other);//矩阵加法,重载运算符"+"
    Matrix operator*(const Matrix& other);//矩阵乘法,重载运算符"*"
    int size(Matrix target);//确定矩阵大小
    Matrix RELU(Matrix origin);//RELU函数
};

#endif