#pragma once
#include "Layer.hpp"
#include <vector>

class Conv2D : public Layer {
private:
    int inChannels, outChannels;
    int inHeight, inWidth;
    int outHeight, outWidth;
    int kernelSize;
    std::vector<std::vector<std::vector<std::vector<double>>>> kernels; 
    std::vector<double> biases;
    std::vector<double> lastInput;
    inline int getIndex(int c, int y, int x, int height, int width) const {
        return c * (height * width) + y * width + x;
    }

public:
    Conv2D(int inC, int inH, int inW, int outC, int kSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
    void save_weights(std::ofstream& file) override;
    void load_weights(std::ifstream& file) override;
};