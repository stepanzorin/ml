#include "mean_absolute_error.hpp"

#include <cassert>
#include <cmath>
#include <ranges>

namespace ml::metrics::regression {

double mean_absolute_error(const std::span<const double> targets, const std::span<const double> predictions) {
    assert(!targets.empty());
    assert(targets.size() == predictions.size());

    auto error_sum = 0.0;

    for (const auto [target, prediction] : std::views::zip(targets, predictions)) {
        error_sum += std::abs(prediction - target);
    }

    return error_sum / static_cast<double>(targets.size());
}

} // namespace ml::metrics::regression