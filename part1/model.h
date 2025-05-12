#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"


class model{
private :
Matrix weight1;
Matrix bias1;
Matrix weight2;
Matrix bias2;

public:
model( Matrix w1, Matrix b1, Matrix w2, Matrix b2){weight1=w1;weight2=w2;bias1=b1;bias2=b2;};
~model(){};
Matrix forward(Matrix input){Matrix result = softmax(RELU(input * weight1 + bias1) * weight2 + bias2);return result;};
};


#endif