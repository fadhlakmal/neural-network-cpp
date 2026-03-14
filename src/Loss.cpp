#include "../include/Loss.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

double Loss::mse_loss(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if(predictions.size() != targets.size()) throw std::invalid_argument("Size mismatch");
    double loss = 0.0;
    for(size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    return loss / predictions.size();
}

std::vector<double> Loss::mse_gradient(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if(predictions.size() != targets.size()) throw std::invalid_argument("Size mismatch");
    std::vector<double> grad(predictions.size());
    for(size_t i = 0; i < predictions.size(); ++i) {
        grad[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
    }
    return grad;
}

double Loss::cross_entropy_loss(const std::vector<double>& logits, const std::vector<double>& targets) {
    if (logits.size() != targets.size()) throw std::invalid_argument("Size mismatch");
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

    double loss = 0.0;
    for(size_t i = 0; i < logits.size(); ++i) {
        loss += targets[i] * std::log(p[i] + 1e-15);
    }
    return -1 * loss;
}

std::vector<double> Loss::cross_entropy_gradient(const std::vector<double>& logits, const std::vector<double>& targets) {
    if (logits.size() != targets.size()) throw std::invalid_argument("Size mismatch");
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

    std::vector<double> grad(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        grad[i] = p[i] - targets[i];
    }
    return grad;
}