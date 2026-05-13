// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace ml::metrics {

struct regression_metrics_s {
    double mae = {};
    double mse = {};
    double rmse = {};
    double r2_score = {};
};

[[nodiscard]] regression_metrics_s evaluate_regression_metrics(std::span<const double> targets,
                                                               std::span<const double> predictions);

struct regression_metric_records_s {
    std::optional<regression_metrics_s> first;
    std::optional<regression_metrics_s> last;

    void print_diff() const noexcept;
};


struct binary_classification_metrics_s {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    double log_loss = 0.0;
};

[[nodiscard]] binary_classification_metrics_s evaluate_binary_classification_metrics(
        std::span<const std::uint32_t> targets,
        std::span<const std::uint32_t> predictions,
        std::span<const double> probabilities);

struct binary_classification_metric_records_s {
    std::optional<binary_classification_metrics_s> first;
    std::optional<binary_classification_metrics_s> last;

    void print_diff() const noexcept;
};

} // namespace ml::metrics