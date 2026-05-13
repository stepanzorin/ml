#include "linear_scoring_model_base.hpp"

#include <algorithm>
#include <print>
#include <ranges>
#include <stdexcept>

namespace ml::models::detail {

double LinearScoringModelBase::score(const std::vector<double> &features) const {
    if (features.size() != m_weights.size()) {
        throw std::runtime_error{"Feature count must match weight count"};
    }

    double prediction = m_bias;

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