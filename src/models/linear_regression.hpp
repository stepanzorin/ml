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

class LinearRegression final : public detail::LinearScoringModelBase {
public:
    explicit LinearRegression(const std::size_t feature_count) : detail::LinearScoringModelBase{feature_count} {}

    void train(const std::vector<regression_sample_s> &samples,
               std::uint32_t epoch_count,
               double learning_rate,
               const regularization::regularization_s &regularization = {});

    [[nodiscard]] double predict(const std::vector<double> &features) const;

    void print_parameters() const noexcept;

private:
    metrics::regression_metric_records_s m_metrics_history;
};

} // namespace ml::models