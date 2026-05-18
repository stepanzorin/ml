#include "knn_regressor.hpp"

#include <algorithm>
#include <cassert>
#include <ranges>

#include "algorithms/neighbors.hpp"

namespace ml::models {

KNNRegressor::KNNRegressor(const std::size_t k) : detail::KNNBase<regression_sample_s>{k} {}

double KNNRegressor::predict(const std::vector<double> &features) const noexcept {
    assert(!features.empty());
    assert(features.size() == m_sample_feature_count);

    const auto neighbors = algorithms::find_k_nearest_neighbors(m_k, std::span{std::as_const(m_samples)}, features);

    const auto sum = std::ranges::fold_left(
            neighbors | std::views::transform([this](const algorithms::neighbor_s &neighbor) {
                return m_samples[neighbor.index].target;
            }),
            0.0,
            std::plus<double>{});

    return sum / static_cast<double>(neighbors.size());
}

metrics::regression_metrics_s KNNRegressor::evaluate(const std::vector<regression_sample_s> &samples) const {
    assert(!samples.empty());

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

} // namespace ml::models