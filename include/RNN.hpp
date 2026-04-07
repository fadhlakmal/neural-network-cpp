#pragma once
#include "Layer.hpp"
#include <vector>

class RNN : public Layer {
private:
    int inputSize, hiddenSize;
    std::vector<std::vector<double>> Wxh, Whh;
    std::vector<double> bh;
    std::vector<std::vector<double>> lastInputs, hiddenStates, zStates;

public:
    RNN(int inSize, int hidSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
    void save_weights(std::ofstream& file) override;
    void load_weights(std::ifstream& file) override;
    void reset_states();

};
