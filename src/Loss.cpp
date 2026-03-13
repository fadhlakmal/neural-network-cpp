#include "../include/Loss.hpp"
#include <stdexcept>

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
