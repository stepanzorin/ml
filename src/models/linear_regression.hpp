// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "metrics/regression/metrics.hpp"
#include "regularization/regularization.hpp"
#include "types.hpp"

namespace ml::models {

class LinearRegression {
public:
    explicit LinearRegression(const std::size_t feature_count) : m_bias{0.0}, m_weights(feature_count, 0.0) {}

    [[nodiscard]] double predict(const std::vector<double> &features) const;

    void train(const std::vector<regression_sample_s> &samples,
               std::uint32_t epoch_count,
               double learning_rate,
               const regularization::regularization_s &regularization = {});

    void print_parameters() const noexcept;

private:
    double m_bias;
    std::vector<double> m_weights;

    metrics::regression::first_and_last_metric_records_s<double> m_metrics_history;
};

} // namespace ml::models