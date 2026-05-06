#include "generalization.hpp"

#include <cmath>
#include <format>
#include <utility>

#include <spdlog/spdlog.h>

namespace ml::metrics {

namespace {

constexpr auto epsilon = 1e-12;


enum class generalization_type_e { underfitting, good, overfitting, suspicious_split, worse_than_baseline };

[[nodiscard]] std::string generalization_type_to_string(const generalization_type_e type) {
    switch (type) {
        case generalization_type_e::underfitting: return "Underfitting";
        case generalization_type_e::good: return "Good";
        case generalization_type_e::overfitting: return "Overfitting";
        case generalization_type_e::suspicious_split: return "Suspicious split";
        case generalization_type_e::worse_than_baseline: return "Worse than baseline";
        default: assert(false);
    }
    std::unreachable();
}


void warn_if_score_looks_suspicious(const std::string &metric_name, const double score, const metric_range_e range) {
    if (!std::isfinite(score)) {
        spdlog::warn("{} score is not finite", metric_name);
        return;
    }

    switch (range) {
        case metric_range_e::zero_to_one:
            if (score < 0.0 || score > 1.0) {
                spdlog::warn("{} score looks suspicious: expected range [0, 1], got {}", metric_name, score);
            }
            break;

        case metric_range_e::zero_to_hundred:
            if (score < 0.0 || score > 100.0) {
                spdlog::warn("{} score looks suspicious: expected range [0, 100], got {}", metric_name, score);
            }
            break;

        case metric_range_e::non_negative:
            if (score < 0.0) {
                spdlog::warn("{} score looks suspicious: expected non-negative value, got {}", metric_name, score);
            }
            break;

        case metric_range_e::unbounded: break;
    }
}


namespace detail {

[[nodiscard]] std::string make_gap_message(std::string &&message, const double relative_gap) {
    return std::format("{}. Relative gap: {:.2f}%", std::move(message), relative_gap * 100.0);
}

} // namespace detail

void check_relative_gap(const double relative_gap,
                        const bool print_debug_info,
                        const double relative_gap_threshold = 0.2) {
    if (print_debug_info) {
        if (relative_gap > relative_gap_threshold) {
            spdlog::warn("{}",
                         detail::make_gap_message("Possible overfitting: holdout score is worse than train score",
                                                  relative_gap));
        } else if (relative_gap < -relative_gap_threshold) {
            spdlog::warn("{}",
                         detail::make_gap_message("Suspicious split: holdout score is better than train score",
                                                  relative_gap));
        } else {
            spdlog::debug("{}", detail::make_gap_message("Generalization looks acceptable", relative_gap));
        }
    }
}

[[nodiscard]] double calculate_absolute_gap(const double train_score,
                                            const double holdout_score,
                                            const metric_direction_e direction) noexcept {
    switch (direction) {
        case metric_direction_e::lower_is_better: return holdout_score - train_score;
        case metric_direction_e::higher_is_better: return train_score - holdout_score;
    }

    std::unreachable();
}

[[nodiscard]] double calculate_relative_gap(const double train_score,
                                            const double holdout_score,
                                            const metric_direction_e direction) noexcept {
    const auto denominator = std::max(std::abs(train_score), epsilon);
    return calculate_absolute_gap(train_score, holdout_score, direction) / denominator;
}

namespace detail {
[[nodiscard]] std::string make_diagnosis_lower_is_better(const double train_score,
                                                         const double holdout_score,
                                                         const double relative_gap,
                                                         const double baseline_score,
                                                         const double relative_gap_threshold = 0.2,
                                                         const double baseline_margin = 0.05) {
    if (!std::isfinite(train_score) || !std::isfinite(holdout_score) || !std::isfinite(baseline_score)) {
        return generalization_type_to_string(generalization_type_e::suspicious_split);
    }

    const auto train_beats_baseline = train_score < baseline_score * (1.0 - baseline_margin);

    const auto holdout_beats_baseline = holdout_score < baseline_score * (1.0 - baseline_margin);

    if (relative_gap < -relative_gap_threshold) {
        return generalization_type_to_string(generalization_type_e::suspicious_split);
    }

    if (!train_beats_baseline && !holdout_beats_baseline) {
        return generalization_type_to_string(generalization_type_e::underfitting);
    }

    if (train_beats_baseline && !holdout_beats_baseline) {
        return generalization_type_to_string(generalization_type_e::overfitting);
    }

    if (relative_gap > relative_gap_threshold) {
        return generalization_type_to_string(generalization_type_e::overfitting);
    }

    if (holdout_beats_baseline) {
        return generalization_type_to_string(generalization_type_e::good);
    }

    return generalization_type_to_string(generalization_type_e::worse_than_baseline);
}

[[nodiscard]] std::string make_diagnosis_higher_is_better(const double train_score,
                                                          const double holdout_score,
                                                          const double relative_gap,
                                                          const double baseline_score,
                                                          const double relative_gap_threshold = 0.2,
                                                          const double baseline_min_delta = 0.05) {
    if (!std::isfinite(train_score) || !std::isfinite(holdout_score) || !std::isfinite(baseline_score)) {
        return generalization_type_to_string(generalization_type_e::suspicious_split);
    }

    const auto train_beats_baseline = train_score > baseline_score + baseline_min_delta;

    const auto holdout_beats_baseline = holdout_score > baseline_score + baseline_min_delta;

    if (relative_gap < -relative_gap_threshold) {
        return generalization_type_to_string(generalization_type_e::suspicious_split);
    }

    if (!train_beats_baseline && !holdout_beats_baseline) {
        return generalization_type_to_string(generalization_type_e::underfitting);
    }

    if (train_beats_baseline && !holdout_beats_baseline) {
        return generalization_type_to_string(generalization_type_e::overfitting);
    }

    if (relative_gap > relative_gap_threshold) {
        return generalization_type_to_string(generalization_type_e::overfitting);
    }

    if (holdout_beats_baseline) {
        return generalization_type_to_string(generalization_type_e::good);
    }

    return generalization_type_to_string(generalization_type_e::worse_than_baseline);
}

} // namespace detail

[[nodiscard]] std::string make_diagnosis(const double train_score,
                                         const double holdout_score,
                                         const double relative_gap,
                                         const double baseline_score,
                                         const metric_direction_e direction,
                                         const double relative_gap_threshold = 0.2) {
    switch (direction) {
        case metric_direction_e::lower_is_better:
            return detail::make_diagnosis_lower_is_better(train_score,
                                                          holdout_score,
                                                          relative_gap,
                                                          baseline_score,
                                                          relative_gap_threshold);

        case metric_direction_e::higher_is_better:
            return detail::make_diagnosis_higher_is_better(train_score,
                                                           holdout_score,
                                                           relative_gap,
                                                           baseline_score,
                                                           relative_gap_threshold);
    }

    std::unreachable();
}

[[nodiscard]] std::string make_diagnosis_without_baseline(const double train_score,
                                                          const double holdout_score,
                                                          const double relative_gap,
                                                          const double relative_gap_threshold = 0.2) {
    if (!std::isfinite(train_score) || !std::isfinite(holdout_score)) {
        return generalization_type_to_string(generalization_type_e::suspicious_split);
    }

    if (relative_gap < -relative_gap_threshold) {
        return generalization_type_to_string(generalization_type_e::suspicious_split);
    }

    if (relative_gap > relative_gap_threshold) {
        return generalization_type_to_string(generalization_type_e::overfitting);
    }

    return generalization_type_to_string(generalization_type_e::good);
}

} // namespace

void generalization_report_s::print() const {
    const auto direction_string = [this] {
        switch (direction) {
            case metric_direction_e::lower_is_better: return "lower is better";
            case metric_direction_e::higher_is_better: return "higher is better";
        }
        std::unreachable();
    };

    const auto baseline_string = baseline_score.has_value() ? std::format("{}", *baseline_score)
                                                            : std::string{"<not provided>"};

    const auto interpretation = [this] {
        if (relative_gap > 0.0) {
            return std::format("{} is worse than train by {:.2f}%", holdout_name, relative_gap * 100.0);
        }

        if (relative_gap < 0.0) {
            return std::format("{} is better than train by {:.2f}%", holdout_name, std::abs(relative_gap) * 100.0);
        }

        return std::format("{} and train scores are equal", holdout_name);
    };

    spdlog::info("Generalization report for metric \"{}\"\n"
                 "    Direction: {}\n"
                 "    Diagnosis: {}\n"
                 "    Scores:\n"
                 "        Train: {}\n"
                 "        {}: {}\n"
                 "        Baseline: {}\n"
                 "    Gap:\n"
                 "        Absolute: {}\n"
                 "        Relative: {:.2f}%\n"
                 "    Interpretation:\n"
                 "        {}",
                 metric_name,
                 direction_string(),
                 diagnosis,
                 train_score,
                 holdout_name,
                 holdout_score,
                 baseline_string,
                 absolute_gap,
                 relative_gap * 100.0,
                 interpretation());
}

generalization_report_s analyze_generalization(const std::string &metric_name,
                                               const double train_score,
                                               const double holdout_score,
                                               const metric_direction_e direction,
                                               const metric_range_e metric_range,
                                               const std::optional<double> custom_baseline_score,
                                               const double relative_gap_threshold,
                                               const bool print_debug_info,
                                               std::string holdout_name) {
    warn_if_score_looks_suspicious(metric_name, train_score, metric_range);
    warn_if_score_looks_suspicious(metric_name, holdout_score, metric_range);

    if (custom_baseline_score.has_value()) {
        warn_if_score_looks_suspicious(metric_name, *custom_baseline_score, metric_range);
    }

    const auto absolute_gap = calculate_absolute_gap(train_score, holdout_score, direction);

    const auto relative_gap = calculate_relative_gap(train_score, holdout_score, direction);

    check_relative_gap(relative_gap, print_debug_info, relative_gap_threshold);

    const auto diagnosis = custom_baseline_score.has_value() ? make_diagnosis(train_score,
                                                                              holdout_score,
                                                                              relative_gap,
                                                                              *custom_baseline_score,
                                                                              direction,
                                                                              relative_gap_threshold)
                                                             : make_diagnosis_without_baseline(train_score,
                                                                                               holdout_score,
                                                                                               relative_gap,
                                                                                               relative_gap_threshold);

    if (!custom_baseline_score.has_value()) {
        spdlog::warn("Baseline score is not provided. Diagnosis will be based only on the train/holdout gap, "
                     "so it cannot confirm whether the model is better than a simple baseline.");
    }

    return {.metric_name = metric_name,
            .diagnosis = diagnosis,
            .train_score = train_score,
            .holdout_score = holdout_score,
            .holdout_name = std::move(holdout_name),
            .baseline_score = custom_baseline_score,
            .absolute_gap = absolute_gap,
            .relative_gap = relative_gap,
            .direction = direction};
}

} // namespace ml::metrics