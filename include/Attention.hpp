#pragma once
#include "Layer.hpp"
#include <vector>
#include <fstream>

class Attention : public Layer {
private:
    int inputDim;
    int headDim;
    int rankDim;
    std::vector<double> W_Q, W_K, W_Vd, W_Vu;
    int lastSeqLen;
    std::vector<double> last_X, last_Q, last_K, last_V, last_P, last_Hv;
    int inline getIndex(int seq_idx, int dim_idx, int matrix_dim) const {
        return seq_idx * matrix_dim + dim_idx;
    }

public:
    Attention(int inputDim, int headDim, int rankDim);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
    void save_weights(std::ofstream& file) override;
    void load_weights(std::ifstream& file) override;

};