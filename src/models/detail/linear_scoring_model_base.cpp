#include "linear_scoring_model_base.hpp"

#include <algorithm>
#include <cassert>
#include <print>
#include <ranges>

namespace ml::models::detail {

double LinearScoringModelBase::score(const std::vector<double> &features) const {
    assert(features.size() == m_weights.size());

    auto prediction = m_bias;

    for (const auto [weight, feature] : std::views::zip(m_weights, features)) {
        prediction += weight * feature;
    }

    return prediction;
}

void LinearScoringModelBase::print_basic_parameters() const noexcept {
    std::println("bias: {}", m_bias);
    std::println("weights: {}\n", m_weights);
}

} // namespace ml::models::detail