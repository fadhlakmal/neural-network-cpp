#include "../include/Linear.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>

Linear::Linear(int inSize, int outSize) : inputSize(inSize), outputSize(outSize) {
    weights.resize(inSize, std::vector<double>(outSize));
    biases.resize(outSize, 0.0);
    
    // he init
    for(int i = 0; i < inSize; ++i) {
        for(int j = 0; j < outSize; ++j) {
            weights[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / inputSize);
        }
    }
}

std::vector<double> Linear::forward(const std::vector<double>& input) {
    lastInput = input;
    std::vector<double> output(outputSize, 0.0);
    for(int j = 0; j < outputSize; ++j) {
        double sum = biases[j];
        for(int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[i][j];
        }
        output[j] = sum;
    }
    return output;
}

std::vector<double> Linear::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> inputGradient(inputSize, 0.0);
    
    for(int i = 0; i < inputSize; ++i) {
        for(int j = 0; j < outputSize; ++j) {
            inputGradient[i] += outputGradient[j] * weights[i][j];
            weights[i][j] -= learningRate * outputGradient[j] * lastInput[i];
        }
    }
    
    for(int j = 0; j < outputSize; ++j) {
        biases[j] -= learningRate * outputGradient[j];
    }
    
    return inputGradient;
}

void Linear::save_weights(std::ofstream& file) {
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
    file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(double));
}

void Linear::load_weights(std::ifstream& file) {
    for (auto& row : weights) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
    file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(double));
}