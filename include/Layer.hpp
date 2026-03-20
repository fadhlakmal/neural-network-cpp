#pragma once
#include <vector>
#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) = 0;

    virtual void save_weights(std::ofstream&) {}
    virtual void load_weights(std::ifstream&) {}
};
