#include "Matrix.h"
#include <vector>
#include <iostream>


//默认构造矩阵
 Matrix::Matrix(int r,int c)
 {
    rows=r;
    colums=c;
    elements.resize(r*c,0.0f);//resize方法用一维向量存储二维数据
 }

 //指定构造矩阵
 Matrix::Matrix(const std::vector<std::vector<float>> mat)
 {
    rows = mat.size();
    colums = mat[0].size();
    elements.resize(rows,colums);
    //循环嵌套结构给矩阵赋值
    for (int i = 0; i <rows; ++i)
    {
        for (int j = 0; j < colums; ++j)
        {
            elements[i*colums+j]=mat[i][j];//一维向量存储二维数据
        }
    }
 }

 Matrix::~Matrix()
{
    //析构函数
}

//矩阵加法
Matrix Matrix::operator+(const Matrix& other)
{
    //判断是否是大小相同的矩阵
    if(rows != other.rows || colums !=other.colums){
        std::cerr <<"不是相同的矩阵，无法相加。" <<std::endl;
        return *this;
    }

    Matrix result(rows,colums);
    
    //两个一维向量相加
    for (int i = 0; i < elements.size(); ++i)
    {
         result.elements[i] =  elements[i]+other.elements[i];
    }
    return result;
}

//重构()进行矩阵索引
float& Matrix:: operator()(int r,int c)
{
    //判断浮点数是否在目标矩阵中
    if(r >= rows || c >= colums){
        std::cerr <<"超出矩阵范围，查找失败" <<std::endl;
    }
    return elements[r*colums+c];
}

//重构const版本的()
const float& Matrix::operator()(int r,int c) const{
    //判断浮点数是否在目标矩阵中
    if(r >= rows || c >= colums){
        std::cerr <<"超出矩阵范围，查找失败" <<std::endl;
    }
    return elements[r*colums+c];
}

//矩阵乘法
Matrix Matrix::operator*(const Matrix& other)
{
    //判断是否是前者列数与后者行数相等的矩阵
    if(colums !=other.rows){
        std::cerr <<"矩阵维度不匹配，无法相乘" <<std::endl;
        return *this;
    }
 
    Matrix result(rows,other.colums);

    //O(n^3)复杂度实现,还需要重构()来进行矩阵索引,other是const类型，因此()需要重构两次
    for(int i = 0;i < rows;++i){
        for(int j = 0;j < other.colums;++j){
            for(int k = 0;k < colums;++k){
                result(i,j)=(*this)(i,k) * other(k,j);
            }
        }
    }
    return result;
}
