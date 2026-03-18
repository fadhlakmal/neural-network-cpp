#pragma once
#include "Layer.hpp"
#include <vector>

class MaxPool2D : public Layer {
private:
    int inChannels;
    int inHeight, inWidth;
    int poolSize;
    int outHeight, outWidth;
    std::vector<int> lastMaxIndices;
    inline int getIndex(int c, int y, int x, int height, int width) const {
        return c * (height * width) + y * width + x;
    }
    
public:
    MaxPool2D(int inC, int inH, int inW, int poolS);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;

};