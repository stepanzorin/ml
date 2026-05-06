// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <optional>
#include <string>

namespace ml::metrics {

enum class metric_direction_e { lower_is_better, higher_is_better };

enum class metric_range_e { unbounded, zero_to_one, zero_to_hundred, non_negative };

struct generalization_report_s {
    std::string metric_name;
    std::string diagnosis = "<unknown>";

    double train_score = 0.0;

    double holdout_score = 0.0;
    std::string holdout_name = "test";

    std::optional<double> baseline_score = std::nullopt;

    double absolute_gap = 0.0;
    double relative_gap = 0.0;

    metric_direction_e direction = metric_direction_e::lower_is_better;

    void print() const;
};

[[nodiscard]] generalization_report_s analyze_generalization(const std::string &metric_name,
                                                             double train_score,
                                                             double holdout_score,
                                                             metric_direction_e direction,
                                                             metric_range_e metric_range,
                                                             std::optional<double> custom_baseline_score = std::nullopt,
                                                             double relative_gap_threshold = 0.2,
                                                             bool print_debug_info = false,
                                                             std::string holdout_name = "test");

} // namespace ml::metrics