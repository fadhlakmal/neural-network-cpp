#include "../include/SelfAttention.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>

SelfAttention::SelfAttention(int inputDim, int headDim, int rankDim) : inputDim(inputDim), headDim(headDim), rankDim(rankDim) {
    int size = inputDim * headDim;
    W_Q.resize(size);
    W_K.resize(size);
    W_Vd.resize(inputDim * rankDim);
    W_Vu.resize(rankDim * headDim);

    for (int i = 0; i < size; ++i) {
        W_Q[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / inputDim);
        W_K[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / inputDim);
    }
    for (int i = 0; i < inputDim * rankDim; ++i) {
        W_Vd[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / inputDim);
    }
    for (int i = 0; i < rankDim * headDim; ++i) {
        W_Vu[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / inputDim);
    }
}

std::vector<double> SelfAttention::forward(const std::vector<double>& input) {
    lastSeqLen = input.size() / inputDim;
    last_X = input;
    last_Q.assign(lastSeqLen * headDim, 0.0);
    last_K.assign(lastSeqLen * headDim, 0.0);
    last_Hv.assign(lastSeqLen * rankDim, 0.0);
    last_V.assign(lastSeqLen * headDim, 0.0);
    last_P.assign(lastSeqLen * lastSeqLen, 0.0);
    std::vector<double> output(lastSeqLen * headDim, 0.0);

    // linear proj
    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double q_sum = 0.0, k_sum = 0.0;
            for (int i = 0; i < inputDim; ++i) {
                q_sum += last_X[getIndex(seq_idx, i, inputDim)] * W_Q[getIndex(i, dim_idx, headDim)];
                k_sum += last_X[getIndex(seq_idx, i, inputDim)] * W_K[getIndex(i, dim_idx, headDim)];
            }
            last_Q[getIndex(seq_idx, dim_idx, headDim)] = q_sum;
            last_K[getIndex(seq_idx, dim_idx, headDim)] = k_sum;
        }

        // down proj
        for (int rank_idx = 0; rank_idx < rankDim; ++rank_idx) {
            double hv_sum = 0.0;
            for (int i = 0; i < inputDim; ++i) {
                hv_sum += input[getIndex(seq_idx, i, inputDim)] * W_Vd[getIndex(i, rank_idx, rankDim)];
            }
            last_Hv[getIndex(seq_idx, rank_idx, rankDim)] = hv_sum;
        }
    }

    // up proj
    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double v_sum = 0.0;
            for (int rank_idx = 0; rank_idx < rankDim; ++rank_idx) {
                v_sum += last_Hv[getIndex(seq_idx, rank_idx, rankDim)] * W_Vu[getIndex(rank_idx, dim_idx, headDim)];
            }
            last_V[getIndex(seq_idx, dim_idx, headDim)] = v_sum;
        }
    }

    // attn
    double scale = std::sqrt(static_cast<double>(headDim));
    for (int q_idx = 0; q_idx < lastSeqLen; ++q_idx) {
        std::vector<double> s_val(lastSeqLen, 0.0);
        double max_s = -1e9;
        for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
            double dot_product = 0.0;
            for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
                dot_product += last_Q[getIndex(q_idx, dim_idx, headDim)] * last_K[getIndex(k_idx, dim_idx, headDim)];
            }
            double s = dot_product / scale;
            s_val[k_idx] = s;
            if (s > max_s) max_s = s;
        }

        double row_sum = 0.0;
        for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
            double p = std::exp(s_val[k_idx] - max_s); 
            last_P[getIndex(q_idx, k_idx, lastSeqLen)] = p;
            row_sum += p;
        }

        for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
            last_P[getIndex(q_idx, k_idx, lastSeqLen)] /= row_sum;
        }
    }

    // output
    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double y = 0.0;
            for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
                y += last_P[getIndex(seq_idx, k_idx, lastSeqLen)] * last_V[getIndex(k_idx, dim_idx, headDim)];
            }
            output[getIndex(seq_idx, dim_idx, headDim)] = y;
        }
    }

    return output;
}

std::vector<double> SelfAttention::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> dW_Q(W_Q.size(), 0.0);
    std::vector<double> dW_K(W_K.size(), 0.0);
    std::vector<double> dW_Vd(W_Vd.size(), 0.0);
    std::vector<double> dW_Vu(W_Vu.size(), 0.0);

    std::vector<double> dV(lastSeqLen * headDim, 0.0);
    std::vector<double> dP(lastSeqLen * lastSeqLen, 0.0);
    std::vector<double> dS(lastSeqLen * lastSeqLen, 0.0);
    std::vector<double> dQ(lastSeqLen * headDim, 0.0);
    std::vector<double> dK(lastSeqLen * headDim, 0.0);
    std::vector<double> dHv(lastSeqLen * rankDim, 0.0);
    std::vector<double> dX(lastSeqLen * inputDim, 0.0);

    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double grad = outputGradient[getIndex(seq_idx, dim_idx, headDim)];
            for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
                double p = last_P[getIndex(seq_idx, k_idx, lastSeqLen)];
                double v = last_V[getIndex(k_idx, dim_idx, headDim)];
                dP[getIndex(seq_idx, k_idx, lastSeqLen)] += grad * v;
                dV[getIndex(k_idx, dim_idx, headDim)] += grad * p;
            }
        }
    }

    // up proj grad
    for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double sum = 0.0;
            for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
                sum += last_Hv[getIndex(seq_idx, r_idx, rankDim)] * dV[getIndex(seq_idx, dim_idx, headDim)];
            }
            dW_Vu[getIndex(r_idx, dim_idx, headDim)] = sum;
        }
    }

    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
            double sum = 0.0;
            for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
                sum += dV[getIndex(seq_idx, dim_idx, headDim)] * W_Vu[getIndex(r_idx, dim_idx, headDim)];
            }
            dHv[getIndex(seq_idx, r_idx, rankDim)] = sum;
        }
    }

    // down proj grad
    for (int i = 0; i < inputDim; ++i) {
        for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
            double sum = 0.0;
            for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
                sum += last_X[getIndex(seq_idx, i, inputDim)] * dHv[getIndex(seq_idx, r_idx, rankDim)];
            }
            dW_Vd[getIndex(i, r_idx, rankDim)] = sum;
        }
    }

    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int i = 0; i < inputDim; ++i) {
            double sum = 0.0;
            for (int r_idx = 0; r_idx < rankDim; ++r_idx) {
                sum += dHv[getIndex(seq_idx, r_idx, rankDim)] * W_Vd[getIndex(i, r_idx, rankDim)];
            }
            dX[getIndex(seq_idx, i, inputDim)] += sum;
        }
    }

    // attn grad
    double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    for (int q_idx = 0; q_idx < lastSeqLen; ++q_idx) {
        double row_sum = 0.0;
        for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
            row_sum += dP[getIndex(q_idx, k_idx, lastSeqLen)] * last_P[getIndex(q_idx, k_idx, lastSeqLen)];
        }
        for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
            double ds_tilde = last_P[getIndex(q_idx, k_idx, lastSeqLen)] * (dP[getIndex(q_idx, k_idx, lastSeqLen)] - row_sum);
            dS[getIndex(q_idx, k_idx, lastSeqLen)] = ds_tilde * scale;
        }
    }

    for (int q_idx = 0; q_idx < lastSeqLen; ++q_idx) {
        for (int k_idx = 0; k_idx < lastSeqLen; ++k_idx) {
            double s_grad = dS[getIndex(q_idx, k_idx, lastSeqLen)];
            for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
                dQ[getIndex(q_idx, dim_idx, headDim)] += s_grad * last_K[getIndex(k_idx, dim_idx, headDim)];
                dK[getIndex(k_idx, dim_idx, headDim)] += s_grad * last_Q[getIndex(q_idx, dim_idx, headDim)];
            }
        }
    }

    for (int seq_idx = 0; seq_idx < lastSeqLen; ++seq_idx) {
        for (int dim_idx = 0; dim_idx < headDim; ++dim_idx) {
            double q_grad = dQ[getIndex(seq_idx, dim_idx, headDim)];
            double k_grad = dK[getIndex(seq_idx, dim_idx, headDim)];
            for (int i = 0; i < inputDim; ++i) {
                dW_Q[getIndex(i, dim_idx, headDim)] += last_X[getIndex(seq_idx, i, inputDim)] * q_grad;
                dW_K[getIndex(i, dim_idx, headDim)] += last_X[getIndex(seq_idx, i, inputDim)] * k_grad;
                dX[getIndex(seq_idx, i, inputDim)] += W_Q[getIndex(i, dim_idx, headDim)] * q_grad + W_K[getIndex(i, dim_idx, headDim)] * k_grad;
            }
        }
    }

    // grad descent
    for (size_t i = 0; i < W_Q.size(); ++i) {
        W_Q[i] -= learningRate * dW_Q[i];
        W_K[i] -= learningRate * dW_K[i];
    }
    for (size_t i = 0; i < W_Vd.size(); ++i) {
        W_Vd[i] -= learningRate * dW_Vd[i];
    }
    for (size_t i = 0; i < W_Vu.size(); ++i) {
        W_Vu[i] -= learningRate * dW_Vu[i];
    }

    return dX;
}

void SelfAttention::save_weights(std::ofstream& file) {
    file.write(reinterpret_cast<const char*>(W_Q.data()), W_Q.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(W_K.data()), W_K.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(W_Vd.data()), W_Vd.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(W_Vu.data()), W_Vu.size() * sizeof(double));
}

void SelfAttention::load_weights(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(W_Q.data()), W_Q.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(W_K.data()), W_K.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(W_Vd.data()), W_Vd.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(W_Vu.data()), W_Vu.size() * sizeof(double));
}