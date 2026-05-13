#include "mean_absolute_error.hpp"

#include <cmath>
#include <ranges>
#include <stdexcept>

namespace ml::metrics::regression {

double mean_absolute_error(const std::span<const double> targets, const std::span<const double> predictions) {
    if (targets.size() != predictions.size()) {
        throw std::runtime_error{"Targets and predictions size mismatch"};
    }

    if (targets.empty()) {
        throw std::runtime_error{"Targets must not be empty"};
    }

    auto error_sum = 0.0;

    for (const auto [target, prediction] : std::views::zip(targets, predictions)) {
        error_sum += std::abs(prediction - target);
    }

    return error_sum / static_cast<double>(targets.size());
}

} // namespace ml::metrics::regression