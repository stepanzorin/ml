#include "linear_regression.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace ml::models {

double LinearRegression::predict(const std::vector<double> &features) {
    if (features.size() != m_weights.size()) {
        throw std::runtime_error{"Feature count must match weight count"};
    }

    double prediction = m_bias;

    for (const auto [weight, feature] : std::views::zip(m_weights, features)) {
        prediction += weight * feature;
    }

    return prediction;
}

void LinearRegression::train(const std::vector<common::sample_s<double>> &samples,
                             const std::uint32_t epoch_count,
                             const double learning_rate) {
    if (samples.empty()) {
        throw std::runtime_error{"Samples must not be empty"};
    }

    if (learning_rate <= 0.0) {
        throw std::runtime_error{"Learning rate must be positive"};
    }

    const auto sample_count = static_cast<double>(samples.size());

    for (const auto _ : std::views::iota(0u, epoch_count)) {
        auto bias_gradient = 0.0;

        auto weight_gradients = std::vector(m_weights.size(), 0.0);

        for (const auto &[features, target] : samples) {
            if (features.size() != m_weights.size()) {
                throw std::runtime_error{"Feature count must match weight count"};
            }

            const auto prediction = predict(features);
            const auto error = prediction - target;

            for (auto &&[weight_gradient, feature] : std::views::zip(weight_gradients, features)) {
                weight_gradient += error * feature;
            }

            bias_gradient += error;
        }

        m_bias -= (bias_gradient / sample_count) * learning_rate;

        for (auto &&[weight, weight_gradient] : std::views::zip(m_weights, weight_gradients)) {
            weight -= (weight_gradient / sample_count) * learning_rate;
        }
    }
}

void LinearRegression::print_parameters() const noexcept {
    std::println("bias: {}", m_bias);
    std::println("weights: {}", m_weights);
}

} // namespace ml::models