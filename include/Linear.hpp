#pragma once
#include "Layer.hpp"
#include <vector>

class Linear : public Layer {
private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> lastInput;

public:
    Linear(int inSize, int outSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
};
