#include "linear_regression.hpp"

#include <algorithm>
#include <cassert>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>

namespace ml::models {

void LinearRegression::train(const std::vector<regression_sample_s> &samples,
                             const std::uint32_t epoch_count,
                             const double learning_rate,
                             const regularization::regularization_s &regularization) {
    if (samples.empty()) {
        throw std::runtime_error{"Samples must not be empty"};
    }

    if (learning_rate <= 0.0) {
        throw std::runtime_error{"Learning rate must be positive"};
    }

    regularization::validate_regularization(regularization);

    m_train_results.first.emplace(evaluate(samples));

    const auto sample_count = static_cast<double>(samples.size());

    for (const auto _ : std::views::iota(0u, epoch_count)) {
        auto bias_gradient = 0.0;

        auto weight_gradients = std::vector(m_weights.size(), 0.0);

        for (const auto &[features, target] : samples) {
            const auto prediction = predict(features);
            const auto error = prediction - target;

            for (auto &&[weight_gradient, feature] : std::views::zip(weight_gradients, features)) {
                weight_gradient += error * feature;
            }

            bias_gradient += error;
        }

        m_bias -= (bias_gradient / sample_count) * learning_rate;

        for (auto &&[weight, weight_gradient] : std::views::zip(m_weights, weight_gradients)) {
            const auto data_gradient = weight_gradient / sample_count;
            const auto penalty_gradient = regularization::regularization_gradient(regularization, weight);
            const auto total_gradient = data_gradient + penalty_gradient;

            if (!std::isfinite(total_gradient)) {
                throw std::runtime_error{"Non-finite gradient detected"};
            }

            weight -= learning_rate * total_gradient;

            if (!std::isfinite(weight)) {
                throw std::runtime_error{"Non-finite weight detected"};
            }
        }
    }

    m_train_results.last.emplace(evaluate(samples));
}

metrics::regression_metrics_s LinearRegression::evaluate(const std::vector<regression_sample_s> &samples) const {
    if (samples.empty()) {
        throw std::runtime_error{"Samples must not be empty"};
    }

    auto predictions = std::vector<double>{};
    auto targets = std::vector<double>{};

    predictions.reserve(samples.size());
    targets.reserve(samples.size());

    for (const auto &[features, target] : samples) {
        predictions.push_back(predict(features));
        targets.push_back(target);
    }

    return metrics::evaluate_regression_metrics(targets, predictions);
}

double LinearRegression::predict(const std::vector<double> &features) const {
    assert(!features.empty());
    return score(features);
}

void LinearRegression::print_parameters() const noexcept {
    print_basic_parameters();
    std::println("train metrics:");
    m_train_results.print_diff();
}

} // namespace ml::models