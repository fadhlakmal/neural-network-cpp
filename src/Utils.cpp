#include "../include/Utils.hpp"

namespace Utils {

    std::vector<double> softmax(const std::vector<double>& logits) {
        std::vector<double> p(logits.size());
        double max_val = *std::max_element(logits.begin(), logits.end());
        
        double sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            p[i] = std::exp(logits[i] - max_val);
            sum += p[i];
        }
        for (size_t i = 0; i < logits.size(); ++i) {
            p[i] /= sum;
        }
        return p;
    }

    std::vector<double> sigmoid(const std::vector<double>& logits) {
        std::vector<double> p(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            p[i] = 1.0 / (1.0 + std::exp(-logits[i]));
        }
        return p;
    }
}