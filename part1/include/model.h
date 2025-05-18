#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"

template<typename T>
class model{
private :
Matrix<T> weight1;
Matrix<T> bias1;
Matrix<T> weight2;
Matrix<T> bias2;

public:
model( Matrix<T> w1, Matrix<T> b1, Matrix<T> w2, Matrix<T> b2){weight1=w1;weight2=w2;bias1=b1;bias2=b2;};
~model(){};
Matrix<T> forward(Matrix<T> input){Matrix<T> result = softmax(RELU(input * weight1 + bias1) * weight2 + bias2);return result;};
};


#endif