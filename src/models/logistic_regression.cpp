#include "logistic_regression.hpp"

#include <algorithm>
#include <cassert>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>

namespace ml::models {

namespace {

[[nodiscard]] double sigmoid(const double z) noexcept {
    if (z >= 0.0) {
        return 1 / (1 + std::exp(-z));
    }

    return std::exp(z) / (1 + std::exp(z));
}

} // namespace

void LogisticRegression::train(const std::vector<binary_classification_sample_s> &samples,
                               const std::uint32_t epoch_count,
                               const double learning_rate,
                               const regularization::regularization_s &regularization,
                               const double threshold) {
    if (samples.empty()) {
        throw std::runtime_error{"Samples must not be empty"};
    }

    if (learning_rate <= 0.0) {
        throw std::runtime_error{"Learning rate must be positive"};
    }

    if (threshold <= 0.0 || threshold >= 1.0) {
        throw std::runtime_error{"Threshold must be between 0 and 1"};
    }

    regularization::validate_regularization(regularization);

    const auto record_metric = [&](std::optional<metrics::binary_classification_metrics_s> &record) {
        auto targets = std::vector<std::uint32_t>{};
        auto predictions = std::vector<std::uint32_t>{};
        auto probabilities = std::vector<double>{};

        targets.reserve(samples.size());
        predictions.reserve(samples.size());
        probabilities.reserve(samples.size());

        for (const auto &[features, label] : samples) {
            if (label > 1) {
                throw std::runtime_error{"Binary classification label must be 0 or 1"};
            }

            const auto probability = predict_probability(features);
            const auto prediction = probability >= threshold ? 1u : 0u;

            targets.push_back(label);
            predictions.push_back(prediction);
            probabilities.push_back(probability);
        }

        record.emplace(metrics::evaluate_binary_classification_metrics(targets, predictions, probabilities));
    };

    record_metric(m_metrics_history.first);

    const auto sample_count = static_cast<double>(samples.size());

    for (const auto _ : std::views::iota(0u, epoch_count)) {
        auto bias_gradient = 0.0;

        auto weight_gradients = std::vector(m_weights.size(), 0.0);

        for (const auto &[features, label] : samples) {
            if (label > 1) {
                throw std::runtime_error{"Binary classification label must be 0 or 1"};
            }

            const auto score = this->score(features);
            const auto probability = sigmoid(score);
            const auto error = probability - static_cast<double>(label);

            if (!std::isfinite(score)) {
                throw std::runtime_error{"Non-finite logistic score detected"};
            }

            if (!std::isfinite(probability)) {
                throw std::runtime_error{"Non-finite probability detected"};
            }

            if (!std::isfinite(error)) {
                throw std::runtime_error{"Non-finite logistic error detected"};
            }

            for (auto &&[weight_gradient, feature] : std::views::zip(weight_gradients, features)) {
                weight_gradient += error * feature;
            }

            bias_gradient += error;
        }

        m_bias -= (bias_gradient / sample_count) * learning_rate;

        if (!std::isfinite(m_bias)) {
            throw std::runtime_error{"Non-finite bias detected"};
        }

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

    record_metric(m_metrics_history.last);
}

double LogisticRegression::linear_score(const std::vector<double> &features) const {
    assert(!features.empty());
    return score(features);
}

std::uint32_t LogisticRegression::predict_label(const std::vector<double> &features, const double threshold) const {
    assert(!features.empty());
    assert(threshold >= 0.0 && threshold < 1.0);
    const auto probability = predict_probability(features);
    return probability >= threshold;
}

double LogisticRegression::predict_probability(const std::vector<double> &features) const {
    assert(!features.empty());
    return sigmoid(linear_score(features));
}

void LogisticRegression::print_parameters() const noexcept {
    print_basic_parameters();
    std::println("train metrics:");
    m_metrics_history.print_diff();
}

} // namespace ml::models