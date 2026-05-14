// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "metrics/metrics.hpp"
#include "models/detail/linear_scoring_model_base.hpp"
#include "regularization/regularization.hpp"

namespace ml::models {

class LogisticRegression final : public detail::LinearScoringModelBase {
public:
    explicit LogisticRegression(const std::size_t feature_count) : detail::LinearScoringModelBase{feature_count} {}

    void train(const std::vector<binary_classification_sample_s> &samples,
               std::uint32_t epoch_count,
               double learning_rate,
               const regularization::regularization_s &regularization = {},
               double threshold = 0.5);

    [[nodiscard]] metrics::binary_classification_metrics_s evaluate(
            const std::vector<binary_classification_sample_s> &samples,
            double threshold = 0.5) const;

    [[nodiscard]] double linear_score(const std::vector<double> &features) const;
    [[nodiscard]] std::uint32_t predict_label(const std::vector<double> &features, double threshold = 0.5) const;
    [[nodiscard]] double predict_probability(const std::vector<double> &features) const;

    void print_parameters() const noexcept;

private:
    metrics::binary_classification_metric_records_s m_train_results;
};

} // namespace ml::models