#pragma once
#ifndef Matrix_H
#define Matrix_H

#include <vector>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

//基于int行列的矩阵无法生成矩阵，矩阵索引超出范围，修改为size_t类型，还是无法正常生成，排除int的问题
//将矩阵索引的运算符重载内的cerr改为throw,还是无法解决问题
//啊，原来是默认构造函数里的行和列写反了，改了还是不行
//原来print()函数里写成了两个++i，行了

// 前向声明Matrix类
template<typename T> class Matrix;

// 前向声明友元函数模板
template<typename T> Matrix<T> RELU(const Matrix<T>&);
template<typename T> Matrix<T> softmax(const Matrix<T>&);

template<typename T> 
class Matrix
{
public:
    std::vector<T> elements;
    size_t rows;
    size_t colums;
    
    Matrix(){rows = 0; colums = 0;};
    Matrix(size_t r,size_t c):rows(r),colums(c),elements(r*c){};//默认构造函数 
    Matrix(const std::vector<std::vector<float>> mat); //指定构造矩阵
    explicit Matrix(const cv::Mat& srcImage);//Opencv矩阵转换构造
    ~Matrix(){};//析构函数

    size_t get_rows() const{return rows;};
    size_t get_colums() const{return colums;};
    
    T& operator()(size_t r,size_t c); //重构()进行矩阵索引
    const T& operator()(size_t r,size_t c) const;//重构const版本的()

    Matrix<T> operator+(const Matrix<T>& other);//矩阵加法,重载运算符"+"
    Matrix<T> operator*(const Matrix<T>& other); //矩阵乘法,重载运算符"*"
    
    void print() const;//打印矩阵  
    
    friend Matrix<T> RELU<>(const Matrix<T>&);//RELU函数
    friend Matrix<T> softmax<>(const Matrix<T>&);//softmax函数
};

//Opencv矩阵转换构造
template<typename T> 
Matrix<T>::Matrix(const cv::Mat& srcImage)
{
    if (srcImage.channels()!=1)
    {
        std::cerr <<"错误的通道数，仅支持灰度图" <<std::endl;
        return;
    }
    size_t l =srcImage.cols;
    size_t r =srcImage.rows;
    //拍扁，每个元素归一化
    rows =1;
    colums = l*r;
    for(size_t i = 0;i<rows;++i){
        for (size_t j = 0; j < colums; ++j){
            uchar pixelValue = srcImage.at<uchar>(i,j);
            float trans = static_cast<float>(pixelValue)/ 255.0f;
            elements.push_back(trans); 
        }
     }
 }

 //重构()进行矩阵索引
 template<typename T>
 T& Matrix<T>::operator()(size_t r,size_t c)
{
    //判断浮点数是否在目标矩阵中
    if(r >= rows || c >= colums){
       throw std::out_of_range("矩阵索引超出范围");
    }
    return elements[r*colums+c];
}

//重构const版本的()
template<typename T>
const T& Matrix<T>::operator()(size_t r,size_t c) const
{
    //判断浮点数是否在目标矩阵中
    if(r >= rows || c >= colums){
        throw std::out_of_range("矩阵索引超出范围");
    }
    return elements[r*colums+c];
}

//矩阵加法,重载运算符"+"
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other)
 {
    //判断是否是大小相同的矩阵
    if(rows != other.rows || colums !=other.colums){
        std::cerr <<"不是相同的矩阵，无法相加。" <<std::endl;
        return *this;
    }

    Matrix<T> result(rows,colums);
    
    //两个一维向量相加
    for (size_t i = 0; i < rows*colums; ++i)
    {
         result.elements[i] =  elements[i]+other.elements[i];
    }
    return result;
}

//矩阵乘法,重载运算符"*"
//part4修改：多线程，每个线程负责计算一部分行列的矩阵相乘，然后求和
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) 
    {
    //判断是否是前者列数与后者行数相等的矩阵
    if(colums !=other.rows){
        std::cerr <<"矩阵维度不匹配，无法相乘" <<std::endl;
        return *this;
    }
 
    Matrix<T> result(rows,other.colums);

    //O(n^3)复杂度实现,还需要重构()来进行矩阵索引,other是const类型，因此()需要重构两次
    for(size_t i = 0;i < rows;++i){
        for(size_t j = 0;j < other.colums;++j){
            for(size_t k = 0;k < colums;++k){
                result(i,j)=result(i,j)+(*this)(i,k) * other(k,j);
            }
        }
    }
    return result;
}

//打印矩阵
template<typename T>
void Matrix<T>::print() const
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < colums; ++j)
        {
            std::cout << elements[i*colums+j] << "\t";
        }
        std::cout<<std::endl;
    }
}

//RELU函数
template<typename T> 
Matrix<T> RELU(const Matrix<T>& target){    
    Matrix<T> result = target;
    for (size_t i = 0; i < result.get_colums()*result.get_rows(); ++i)
    {
        if(result.elements[i] < 0){
        result.elements[i] = 0;
        }
    }
    return result;
}

//SoftMax函数
template<typename T> 
Matrix<T> softmax(const Matrix<T>& target)
{
    //行向量情况
    if(target.get_colums() == 1 && target.get_rows() != 1){
        
        float sum=0;
        size_t x=target.get_rows();
        Matrix<T> e_num(x,1);
        Matrix<T> result(x,1);
        
        for (size_t i = 0; i < x ; ++i)
        {
            e_num.elements[i]=std::exp(target.elements[i]);
            sum += e_num.elements[i];
        }
        for (size_t i = 0; i < x; ++i)
        {
            result.elements[i] = e_num.elements[i]/sum;
        }
        return result;
    }
    //列向量情况
    else if (target.get_rows() == 1 && target.get_colums() != 1)
    {
        float sum=0;
        size_t x=target.get_colums();
        Matrix<T> e_num(1,x);
        Matrix<T> result(1,x);
        
        for (size_t i = 0; i < x ; ++i)
        {
            e_num.elements[i]=std::exp(target.elements[i]);
            sum += e_num.elements[i];
        }
        for (size_t i = 0; i < x; ++i)
        {
            result.elements[i] = e_num.elements[i]/sum;
        }
        return result;
    }
    else{
        std::cerr <<"这不是一个向量，无法进行操作" <<std::endl;
        return Matrix<T>();
    }
}

#endif