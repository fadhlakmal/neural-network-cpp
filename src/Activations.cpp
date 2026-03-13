#include "../include/Activations.hpp"
#include <cmath>

std::vector<double> ReLU::forward(const std::vector<double>& input) {
    lastInput = input;
    std::vector<double> output(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
    return output;
}

std::vector<double> ReLU::backward(const std::vector<double>& outputGradient, double learningRate) {
    (void)learningRate;
    std::vector<double> inputGradient(outputGradient.size());
    for(size_t i = 0; i < outputGradient.size(); ++i) {
        inputGradient[i] = lastInput[i] > 0 ? outputGradient[i] : 0;
    }
    return inputGradient;
}

std::vector<double> Sigmoid::forward(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0 / (1.0 + std::exp(-input[i]));
    }
    lastOutput = output;
    return output;
}

std::vector<double> Sigmoid::backward(const std::vector<double>& outputGradient, double learningRate) {
    (void)learningRate;
    std::vector<double> inputGradient(outputGradient.size());
    for(size_t i = 0; i < outputGradient.size(); ++i) {
        inputGradient[i] = outputGradient[i] * lastOutput[i] * (1.0 - lastOutput[i]);
    }
    return inputGradient;
}

std::vector<double> Tanh::forward(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    lastOutput = output;
    return output;
}

std::vector<double> Tanh::backward(const std::vector<double>& outputGradient, double learningRate) {
    (void)learningRate;
    std::vector<double> inputGradient(outputGradient.size());
    for(size_t i = 0; i < outputGradient.size(); ++i) {
        inputGradient[i] = outputGradient[i] * (1.0 - lastOutput[i] * lastOutput[i]);
    }
    return inputGradient;
}
