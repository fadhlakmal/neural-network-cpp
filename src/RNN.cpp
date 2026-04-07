#include "../include/RNN.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>

RNN::RNN(int inSize, int hidSize) : inputSize(inSize), hiddenSize(hidSize) {
    Wxh.resize(inputSize, std::vector<double>(hiddenSize));
    Whh.resize(hiddenSize, std::vector<double>(hiddenSize));
    bh.resize(hiddenSize, 0.0);

    // xavier init
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            Wxh[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(1.0 / inputSize);
        }
    }
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            Whh[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(1.0 / hiddenSize);
        }
    }
    
}

std::vector<double> RNN::forward(const std::vector<double>& input) {
    lastInputs.push_back(input);
    if (hiddenStates.empty()) hiddenStates.push_back(std::vector<double>(hiddenSize, 0.0));
    std::vector<double> z(hiddenSize, 0.0);
    std::vector<double> h(hiddenSize, 0.0);
    
    for (int hi = 0; hi < hiddenSize; ++hi) {
        for (int i = 0; i < inputSize; ++i) {
            z[hi] += lastInputs.back()[i] * Wxh[i][hi];
        }
        for (int j = 0; j < hiddenSize; ++j) {
            z[hi] += hiddenStates.back()[j] * Whh[j][hi];
        }
        z[hi] += bh[hi];
        h[hi] = std::tanh(z[hi]);
    }

    zStates.push_back(z);
    hiddenStates.push_back(h);

    return h; 
}

std::vector<double> RNN::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> inputGradient(inputSize, 0.0);
    std::vector<double> hiddenGradient = outputGradient;

    for (int t = hiddenStates.size() - 1; t > 0; --t) {
        std::vector<double> nextHiddenGradient(hiddenSize, 0.0);
        for (int hi = 0; hi < hiddenSize; ++hi) {
            double dz = hiddenGradient[hi] * (1 - std::tanh(zStates[t-1][hi]) * std::tanh(zStates[t-1][hi]));

            for (int in = 0; in < inputSize; ++in) {
                inputGradient[in] += Wxh[in][hi] * dz;
                Wxh[in][hi] -= learningRate * dz * lastInputs[t-1][in];
            }
            for (int hj = 0; hj < hiddenSize; ++hj) {
                nextHiddenGradient[hj] += Whh[hj][hi] * dz;
                Whh[hj][hi] -= learningRate * dz * hiddenStates[t-1][hj];
            }
            bh[hi] -= learningRate * dz;
        }
        hiddenGradient = nextHiddenGradient;
    }

    return inputGradient;
}

void RNN::save_weights(std::ofstream& file) {
    for (const auto& row : Wxh) file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    for (const auto& row : Whh) file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(bh.data()), bh.size() * sizeof(double));
}

void RNN::load_weights(std::ifstream& file) {
    for (auto& row : Wxh) file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    for (auto& row : Whh) file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(bh.data()), bh.size() * sizeof(double));
}

void RNN::reset_states() {
    lastInputs.clear();
    hiddenStates.clear();
    zStates.clear();
}