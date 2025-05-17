#include "Matrix.h"
#include <vector>
#include <iostream>
#include <cmath>

//默认构造矩阵
 Matrix::Matrix(size_t r,size_t c)
 {
    rows=r;
    colums=c;
    elements.resize(rows*colums,0.0f);//resize方法用一维向量存储二维数据
 }

 //指定构造矩阵
 Matrix::Matrix(const std::vector<std::vector<float>> mat)
 {
    if (mat.empty()) {
        rows = colums = 0;
        return;
    }
    rows = mat.size();
    colums = mat[0].size();
    elements.resize(rows*colums);
    //循环嵌套结构给矩阵赋值
    for (size_t i = 0; i <rows; ++i)
    {
        for (size_t j = 0; j < colums; ++j)
        {
            elements[i*colums+j]=mat[i][j];//一维向量存储二维数据
        }
    }
 }

 //OpenCV矩阵转换
Matrix::Matrix(const cv::Mat& srcImage){
        if (srcImage.channels()!=1)
        {
            std::cerr <<"错误的通道数，仅支持灰度图" <<std::endl;
            return;
        }
        colums =srcImage.cols;
        rows =srcImage.rows;
        elements.resize(rows*colums,0.0f);//resize方法用一维向量存储二维数据
        //每个元素归一化
        for(size_t i = 0;i<rows;++i){
             for (size_t j = 0; j < colums; ++j){
                uchar pixelValue = srcImage.at<uchar>(i,j);
                elements[i*colums+j] = static_cast<float>(pixelValue)/ 255.0f;
             }
        }
 }

 Matrix::~Matrix()
{
    //析构函数
}

//打印矩阵
void Matrix::print() const
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
    for (size_t i = 0; i < rows*colums; ++i)
    {
         result.elements[i] =  elements[i]+other.elements[i];
    }
    return result;
}

//重构()进行矩阵索引
float& Matrix:: operator()(size_t r,size_t c)
{
    //判断浮点数是否在目标矩阵中
    if(r >= rows || c >= colums){
       throw std::out_of_range("矩阵索引超出范围");
    }
    return elements[r*colums+c];
}

//重构const版本的()
const float& Matrix::operator()(size_t r,size_t c) const{
    //判断浮点数是否在目标矩阵中
    if(r >= rows || c >= colums){
        throw std::out_of_range("矩阵索引超出范围");
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
    for(size_t i = 0;i < rows;++i){
        for(size_t j = 0;j < other.colums;++j){
            for(size_t k = 0;k < colums;++k){
                result(i,j)=result(i,j)+(*this)(i,k) * other(k,j);
            }
        }
    }
    return result;
}

//RELU函数
Matrix RELU(Matrix target){    
    Matrix result = target;
    for (size_t i = 0; i < result.get_colums()*result.get_rows(); ++i)
    {
        if(result.elements[i] < 0){
        result.elements[i] = 0;
        }
    }
    return result;
}

//SoftMax函数
Matrix softmax(Matrix target)
{
    //行向量情况
    if(target.get_colums() == 1 && target.get_rows() != 1){
        
        float sum=0.0f;
        size_t x=target.get_rows();
        Matrix e_num(x,1);
        Matrix result(x,1);
        
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
        float sum=0.0f;
        size_t x=target.get_colums();
        Matrix e_num(1,x);
        Matrix result(1,x);
        
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
        return Matrix();
    }
}


