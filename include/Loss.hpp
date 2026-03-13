#pragma once
#include <vector>

class Loss {
public:
    static double mse_loss(const std::vector<double>& predictions, const std::vector<double>& targets);
    static std::vector<double> mse_gradient(const std::vector<double>& predictions, const std::vector<double>& targets);
};
