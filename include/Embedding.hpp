#pragma once
#include "Layer.hpp"
#include <vector>

class Embedding : public Layer {
private:
    int vocabSize;
    int embeddingDim;
    std::vector<std::vector<double>> embeddings;
    std::vector<int> lastInputIndices;
    inline int getIndex(int vocab_idx, int dim_idx) const {
        return vocab_idx * embeddingDim + dim_idx;
    }

public:
    Embedding(int vocabSize, int embeddingDim);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
    void save_weights(std::ofstream& file) override;
    void load_weights(std::ifstream& file) override;

};