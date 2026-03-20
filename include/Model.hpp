#pragma once
#include <vector>
#include <string>
#include "Layer.hpp"

class Model {
private:
    std::vector<Layer*> layers;
    
public:
    ~Model();
    void add(Layer* layer);
    std::vector<double> forward(std::vector<double> input);
    void backward(std::vector<double> gradient, double learningRate);
    void save(const std::string& filename);
    void load(const std::string& filename);
};
