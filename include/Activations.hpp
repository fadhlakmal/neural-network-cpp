#pragma once
#include "Layer.hpp"
#include <vector>

class ReLU : public Layer {
private:
    std::vector<double> lastInput;
public:
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
};

class Sigmoid : public Layer {
private:
    std::vector<double> lastOutput;
public:
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
};

class Tanh : public Layer {
private:
    std::vector<double> lastOutput;
public:
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
};
