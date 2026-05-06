#include "linear_regression.hpp"

#include <algorithm>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>

#include "metrics/regression/metrics.hpp"

namespace ml::models {

double LinearRegression::predict(const std::vector<double> &features) const {
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

    const auto record_metric = [&](std::optional<metrics::regression::metrics_s<double>> &record) {
        auto predictions = std::vector<double>{};
        auto targets = std::vector<double>{};

        predictions.reserve(samples.size());
        targets.reserve(samples.size());

        for (const auto &[features, target] : samples) {
            predictions.push_back(predict(features));
            targets.push_back(target);
        }

        record.emplace(metrics::regression::evaluate_metrics(targets, predictions));
    };

    record_metric(m_metrics_history.first_train_record);

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
            weight -= (weight_gradient / sample_count) * learning_rate;
        }
    }

    record_metric(m_metrics_history.last_train_record);
}

void LinearRegression::print_parameters() const noexcept {
    std::println("bias: {}", m_bias);
    std::println("weights: {}", m_weights);

    std::println("train metrics:");
    m_metrics_history.print_diff();
}

} // namespace ml::models