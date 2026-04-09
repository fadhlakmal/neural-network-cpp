#include "../include/Embedding.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>

Embedding::Embedding(int vocabSize, int embeddingDim) : vocabSize(vocabSize), embeddingDim(embeddingDim) {
    embeddings.resize(vocabSize, std::vector<double>(embeddingDim));

    for(int i = 0; i < vocabSize; ++i) {
        for(int j = 0; j < embeddingDim; ++j) {
            embeddings[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / embeddingDim);
        }
    }
}

std::vector<double> Embedding::forward(const std::vector<double>& input) {
    lastInputIndices.clear();
    std::vector<double> output(input.size() * embeddingDim, 0.0);

    for (size_t seq_idx = 0; seq_idx < input.size(); ++seq_idx) {
        int vocab_idx = static_cast<int>(input[seq_idx]);
        lastInputIndices.push_back(vocab_idx);
        for (int j = 0; j < embeddingDim; ++j) {
            output[getIndex(seq_idx, j)] = embeddings[vocab_idx][j];
        }
    }
    
    return output;
}

std::vector<double> Embedding::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> inputGradient(lastInputIndices.size(), 0.0);

    for (size_t seq_idx = 0; seq_idx < lastInputIndices.size(); ++seq_idx) {
        int vocab_idx = lastInputIndices[seq_idx];
        for (int j = 0; j < embeddingDim; ++j) {
            double grad = outputGradient[getIndex(seq_idx, j)];
            embeddings[vocab_idx][j] -= learningRate * grad;
        }
    }

    return inputGradient;
}

void Embedding::save_weights(std::ofstream& file) {
    for (const auto& row : embeddings) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
}

void Embedding::load_weights(std::ifstream& file) {
    for (auto& row : embeddings) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
}