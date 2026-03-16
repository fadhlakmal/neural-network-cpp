#include "../include/Loss.hpp"
#include "../include/Utils.hpp"
#include <stdexcept>
#include <algorithm>

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
    std::vector<double> p = Utils::softmax(logits);

    double loss = 0.0;
    for(size_t i = 0; i < logits.size(); ++i) {
        loss += targets[i] * Utils::safelog(p[i]);
    }
    return -1 * loss;
}

std::vector<double> Loss::cross_entropy_gradient(const std::vector<double>& logits, const std::vector<double>& targets) {
    if (logits.size() != targets.size()) throw std::invalid_argument("Size mismatch");
    std::vector<double> p = Utils::softmax(logits);

    std::vector<double> grad(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        grad[i] = p[i] - targets[i];
    }
    return grad;
}

double Loss::binary_cross_entropy_loss(const std::vector<double>& logits, const std::vector<double>& targets) {
    if (logits.size() != targets.size()) throw std::invalid_argument("Size mismatch");
    std::vector<double> p = Utils::sigmoid(logits);

    double loss = 0.0;
    for(size_t i = 0; i < logits.size(); ++i) {
        loss += targets[i] * Utils::safelog(p[i]) + (1 - targets[i]) * Utils::safelog(1 - p[i]);
    }
    return (-1.0 * loss) / logits.size();
}

std::vector<double> Loss::binary_cross_entropy_gradient(const std::vector<double>& logits, const std::vector<double>& targets) {
    if (logits.size() != targets.size()) throw std::invalid_argument("Size mismatch");
    std::vector<double> p = Utils::sigmoid(logits);

    std::vector<double> grad(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        grad[i] = (p[i] - targets[i]) / logits.size();
    }
    return grad;
}
