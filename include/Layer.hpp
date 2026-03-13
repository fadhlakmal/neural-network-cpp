#pragma once
#include <vector>

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) = 0;
};
