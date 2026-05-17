#include "euclidean_distance.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace ml::algorithms {

double compute_squared_euclidean_distance(const std::span<const double> lhs, const std::span<const double> rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error{"Failed to compute squared Euclidean distance: feature vector size mismatch"};
    }

    auto sum = 0.0;

    for (const auto [left, right] : std::views::zip(lhs, rhs)) {
        const auto diff = left - right;
        sum += diff * diff;
    }

    return sum;
}

} // namespace ml::algorithms