#include "../include/CrossAttention.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <stdexcept>

CrossAttention::CrossAttention(int queryDim, int contextDim, int headDim, int rankDim) : queryDim(queryDim), contextDim(contextDim), headDim(headDim), rankDim(rankDim), querySeqLen(0), contextSeqLen(0) {
    
    W_Q.resize(queryDim * headDim);
    W_K.resize(contextDim * headDim);
    W_Vd.resize(contextDim * rankDim);
    W_Vu.resize(rankDim * headDim);

    for (int i = 0; i < queryDim * headDim; ++i) {
        W_Q[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / queryDim);
    }
    for (int i = 0; i < contextDim * headDim; ++i) {
        W_K[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / contextDim);
    }
    for (int i = 0; i < contextDim * rankDim; ++i) {
        W_Vd[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / contextDim);
    }
    for (int i = 0; i < rankDim * headDim; ++i) {
        W_Vu[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / rankDim);
    }
}

void CrossAttention::set_context(const std::vector<double>& context_features, int seq_len) {
    last_context_X = context_features;
    contextSeqLen = seq_len;
    
    if (context_features.size() != static_cast<size_t>(contextSeqLen * contextDim)) {
        throw std::invalid_argument("Context features size mismatch.");
    }
}

std::vector<double> CrossAttention::forward(const std::vector<double>& query_input) {
    querySeqLen = query_input.size() / queryDim;
    last_query_X = query_input;
    
    last_Q.assign(querySeqLen * headDim, 0.0);
    last_K.assign(contextSeqLen * headDim, 0.0);
    last_Hv.assign(contextSeqLen * rankDim, 0.0);
    last_V.assign(contextSeqLen * headDim, 0.0);
    last_P.assign(querySeqLen * contextSeqLen, 0.0);
    
    std::vector<double> output(querySeqLen * headDim, 0.0);

    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double q_sum = 0.0;
            for (int i = 0; i < queryDim; ++i) {
                q_sum += last_query_X[getIndex(q_idx, i, queryDim)] * W_Q[getIndex(i, dim_idx, headDim)];
            }
            last_Q[getIndex(q_idx, dim_idx, headDim)] = q_sum;
        }
    }

    // down proj
    for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double k_sum = 0.0;
            for (int i = 0; i < contextDim; ++i) {
                k_sum += last_context_X[getIndex(k_idx, i, contextDim)] * W_K[getIndex(i, dim_idx, headDim)];
            }
            last_K[getIndex(k_idx, dim_idx, headDim)] = k_sum;
        }
        
        for (int rank_idx = 0; rank_idx < rankDim; ++rank_idx) {
            double hv_sum = 0.0;
            for (int i = 0; i < contextDim; ++i) {
                hv_sum += last_context_X[getIndex(k_idx, i, contextDim)] * W_Vd[getIndex(i, rank_idx, rankDim)];
            }
            last_Hv[getIndex(k_idx, rank_idx, rankDim)] = hv_sum;
        }
    }

    // up proj
    for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double v_sum = 0.0;
            for (int rank_idx = 0; rank_idx < rankDim; ++rank_idx) {
                v_sum += last_Hv[getIndex(k_idx, rank_idx, rankDim)] * W_Vu[getIndex(rank_idx, dim_idx, headDim)];
            }
            last_V[getIndex(k_idx, dim_idx, headDim)] = v_sum;
        }
    }

    // attn
    double scale = std::sqrt(static_cast<double>(headDim));
    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        std::vector<double> s_val(contextSeqLen, 0.0);
        double max_s = -1e9;
        
        for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
            double dot_product = 0.0;
            for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
                dot_product += last_Q[getIndex(q_idx, dim_idx, headDim)] * last_K[getIndex(k_idx, dim_idx, headDim)];
            }
            double s = dot_product / scale;
            s_val[k_idx] = s;
            if (s > max_s) max_s = s;
        }

        double row_sum = 0.0;
        for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
            double p = std::exp(s_val[k_idx] - max_s); 
            last_P[getIndex(q_idx, k_idx, contextSeqLen)] = p;
            row_sum += p;
        }

        for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
            last_P[getIndex(q_idx, k_idx, contextSeqLen)] /= row_sum;
        }
    }

    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double y = 0.0;
            for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
                y += last_P[getIndex(q_idx, k_idx, contextSeqLen)] * last_V[getIndex(k_idx, dim_idx, headDim)];
            }
            output[getIndex(q_idx, dim_idx, headDim)] = y;
        }
    }

    return output;
}

std::vector<double> CrossAttention::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> dW_Q(W_Q.size(), 0.0);
    std::vector<double> dW_K(W_K.size(), 0.0);
    std::vector<double> dW_Vd(W_Vd.size(), 0.0);
    std::vector<double> dW_Vu(W_Vu.size(), 0.0);

    std::vector<double> dV(contextSeqLen * headDim, 0.0);
    std::vector<double> dP(querySeqLen * contextSeqLen, 0.0);
    std::vector<double> dS(querySeqLen * contextSeqLen, 0.0);
    std::vector<double> dQ(querySeqLen * headDim, 0.0);
    std::vector<double> dK(contextSeqLen * headDim, 0.0);
    std::vector<double> dHv(contextSeqLen * rankDim, 0.0);
    std::vector<double> dX_query(querySeqLen * queryDim, 0.0);

    // output grad wrt P and V
    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double grad = outputGradient[getIndex(q_idx, dim_idx, headDim)];
            for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
                double p = last_P[getIndex(q_idx, k_idx, contextSeqLen)];
                double v = last_V[getIndex(k_idx, dim_idx, headDim)];
                dP[getIndex(q_idx, k_idx, contextSeqLen)] += grad * v;
                dV[getIndex(k_idx, dim_idx, headDim)] += grad * p;
            }
        }
    }

    // V up proj grad
    for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double sum = 0.0;
            for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
                sum += last_Hv[getIndex(k_idx, r_idx, rankDim)] * dV[getIndex(k_idx, dim_idx, headDim)];
            }
            dW_Vu[getIndex(r_idx, dim_idx, headDim)] = sum;
        }
    }

    for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
        for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
            double sum = 0.0;
            for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
                sum += dV[getIndex(k_idx, dim_idx, headDim)] * W_Vu[getIndex(r_idx, dim_idx, headDim)];
            }
            dHv[getIndex(k_idx, r_idx, rankDim)] = sum;
        }
    }

    // V down proj grad
    for (int i = 0; i < contextDim; ++i) {
        for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
            double sum = 0.0;
            for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
                sum += last_context_X[getIndex(k_idx, i, contextDim)] * dHv[getIndex(k_idx, r_idx, rankDim)];
            }
            dW_Vd[getIndex(i, r_idx, rankDim)] = sum;
        }
    }

    // softmax grad
    double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        double row_sum = 0.0;
        for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
            row_sum += dP[getIndex(q_idx, k_idx, contextSeqLen)] * last_P[getIndex(q_idx, k_idx, contextSeqLen)];
        }
        for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
            double ds_tilde = last_P[getIndex(q_idx, k_idx, contextSeqLen)] * (dP[getIndex(q_idx, k_idx, contextSeqLen)] - row_sum);
            dS[getIndex(q_idx, k_idx, contextSeqLen)] = ds_tilde * scale;
        }
    }

    // S grad wrt Q and K
    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
            double s_grad = dS[getIndex(q_idx, k_idx, contextSeqLen)];
            for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
                dQ[getIndex(q_idx, dim_idx, headDim)] += s_grad * last_K[getIndex(k_idx, dim_idx, headDim)];
                dK[getIndex(k_idx, dim_idx, headDim)] += s_grad * last_Q[getIndex(q_idx, dim_idx, headDim)];
            }
        }
    }

    // Q grad wrt W_Q and query input
    for (int q_idx = 0; q_idx < querySeqLen; ++q_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double q_grad = dQ[getIndex(q_idx, dim_idx, headDim)];
            for (int i = 0; i < queryDim; ++i) {
                dW_Q[getIndex(i, dim_idx, headDim)] += last_query_X[getIndex(q_idx, i, queryDim)] * q_grad;
                dX_query[getIndex(q_idx, i, queryDim)] += W_Q[getIndex(i, dim_idx, headDim)] * q_grad;
            }
        }
    }

    // K grad wrt W_K
    for (int k_idx = 0; k_idx < contextSeqLen; ++k_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double k_grad = dK[getIndex(k_idx, dim_idx, headDim)];
            for (int i = 0; i < contextDim; ++i) {
                dW_K[getIndex(i, dim_idx, headDim)] += last_context_X[getIndex(k_idx, i, contextDim)] * k_grad;
            }
        }
    }

    // grad descent
    for (size_t i = 0; i < W_Q.size(); ++i) W_Q[i] -= learningRate * dW_Q[i];
    for (size_t i = 0; i < W_K.size(); ++i) W_K[i] -= learningRate * dW_K[i];
    for (size_t i = 0; i < W_Vd.size(); ++i) W_Vd[i] -= learningRate * dW_Vd[i];
    for (size_t i = 0; i < W_Vu.size(); ++i) W_Vu[i] -= learningRate * dW_Vu[i];

    return dX_query;
}

void CrossAttention::save_weights(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(W_Q.data()), W_Q.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(W_K.data()), W_K.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(W_Vd.data()), W_Vd.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(W_Vu.data()), W_Vu.size() * sizeof(double));
}

void CrossAttention::load_weights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(W_Q.data()), W_Q.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(W_K.data()), W_K.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(W_Vd.data()), W_Vd.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(W_Vu.data()), W_Vu.size() * sizeof(double));
}