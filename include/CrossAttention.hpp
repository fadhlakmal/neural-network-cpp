#pragma once
#include "Layer.hpp"
#include <vector>
#include <fstream>

class CrossAttention : public Layer {
private:
    int queryDim;
    int contextDim;
    int headDim;
    int rankDim;
    
    std::vector<double> W_Q, W_K, W_Vd, W_Vu;
    
    int querySeqLen;
    int contextSeqLen;
    
    std::vector<double> last_query_X, last_context_X, last_dX_context;
    std::vector<double> last_Q, last_K, last_V, last_P, last_Hv;

    int inline getIndex(int seq_idx, int dim_idx, int matrix_dim) const {
        return seq_idx * matrix_dim + dim_idx;
    }

public:
    CrossAttention(int queryDim, int contextDim, int headDim, int rankDim);
    void set_context(const std::vector<double>& context_features, int seq_len);
    std::vector<double> get_context_gradient();
    std::vector<double> forward(const std::vector<double>& query_input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
    void save_weights(std::ofstream& file) override;
    void load_weights(std::ifstream& file) override;
};