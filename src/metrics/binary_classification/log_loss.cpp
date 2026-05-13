#include "log_loss.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace ml::metrics::binary_classification {

namespace {
constexpr auto epsilon = 1e-15;
}

double evaluate_log_loss(const std::span<const std::uint32_t> targets, const std::span<const double> probabilities) {
    if (targets.size() != probabilities.size()) {
        throw std::runtime_error{"Targets and probabilities size mismatch"};
    }

    if (targets.empty()) {
        throw std::runtime_error{"Cannot calculate log loss for empty dataset"};
    }

    double total_loss = 0.0;

    for (const auto &&[target, probability] : std::views::zip(targets, probabilities)) {
        if (target > 1) {
            throw std::runtime_error{"Binary classification target must be 0 or 1"};
        }

        if (!std::isfinite(probability)) {
            throw std::runtime_error{"Probability must be finite"};
        }

        if (probability < 0.0 || probability > 1.0) {
            throw std::runtime_error{"Probability must be in range [0, 1]"};
        }

        const auto p = std::clamp(probability, epsilon, 1.0 - epsilon);

        const auto y = static_cast<double>(target);

        const auto sample_loss = -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));

        total_loss += sample_loss;
    }

    return total_loss / static_cast<double>(targets.size());
}

} // namespace ml::metrics::binary_classification