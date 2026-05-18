#include "log_loss.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ranges>

namespace ml::metrics::binary_classification {

namespace {

constexpr auto epsilon = 1e-15;

} // namespace

double evaluate_log_loss(const std::span<const std::uint32_t> targets, const std::span<const double> probabilities) {
    assert(!targets.empty());
    assert(targets.size() == probabilities.size());

    auto total_loss = 0.0;

    for (const auto &&[target, probability] : std::views::zip(targets, probabilities)) {
        assert(target <= 1);
        assert(std::isfinite(probability));
        assert(probability >= 0.0 && probability <= 1.0);

        const auto p = std::clamp(probability, epsilon, 1.0 - epsilon);

        const auto y = static_cast<double>(target);

        const auto sample_loss = -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));

        total_loss += sample_loss;
    }

    return total_loss / static_cast<double>(targets.size());
}

} // namespace ml::metrics::binary_classification