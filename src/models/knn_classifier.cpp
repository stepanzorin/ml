#include "knn_classifier.hpp"

#include <algorithm>
#include <cassert>
#include <ranges>

#include "algorithms/neighbors.hpp"

#include <stdexcept>

namespace ml::models {

KNNClassifier::KNNClassifier(const std::size_t k) : detail::KNNBase<binary_classification_sample_s>{k} {}

double KNNClassifier::predict_probability(const std::vector<double> &features) const noexcept {
    assert(!features.empty());
    assert(features.size() == m_sample_feature_count);

    const auto neighbors = algorithms::find_k_nearest_neighbors(m_k, std::span{std::as_const(m_samples)}, features);

    const auto sum = std::ranges::fold_left(
            neighbors | std::views::transform([this](const algorithms::neighbor_s &neighbor) {
                return m_samples[neighbor.index].label;
            }),
            0.0,
            std::plus<double>{});

    return sum / static_cast<double>(neighbors.size());
}

std::uint32_t KNNClassifier::predict_label(const std::vector<double> &features, const double threshold) const noexcept {
    assert(threshold > 0.0 && threshold < 1.0);
    const auto probability = predict_probability(features);
    return probability >= threshold ? 1 : 0;
}

metrics::binary_classification_metrics_s KNNClassifier::evaluate(
        const std::vector<binary_classification_sample_s> &samples,
        const double threshold) const {
    assert(!samples.empty());

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

    return metrics::evaluate_binary_classification_metrics(targets, predictions, probabilities);
}

} // namespace ml::models