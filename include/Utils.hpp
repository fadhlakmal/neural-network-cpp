#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace Utils {
    inline double safelog(double x) {
        return std::log(x + 1e-15);
    }
    std::vector<double> softmax(const std::vector<double>& logits);
    std::vector<double> sigmoid(const std::vector<double>& logits);
}