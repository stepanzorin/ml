// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cassert>
#include <concepts>
#include <vector>
// #include <optional>
#include <print>

#include "common/types.hpp"
#include "metrics/regression/mean_absolute_error.hpp"
#include "metrics/regression/mean_squared_error.hpp"
#include "metrics/regression/r2_score.hpp"
#include "metrics/regression/root_mean_squared_error.hpp"

namespace ml::metrics::regression {

template<std::convertible_to<double> MetricsType>
struct metrics_s {
    MetricsType mae = {};
    MetricsType mse = {};
    MetricsType rmse = {};
    MetricsType r2_score = {};

    void print() const noexcept {
        std::println("mae: {}", mae);
        std::println("mse: {}", mse);
        std::println("rmse: {}", rmse);
        std::println("r2_score: {}\n", r2_score);
    }
};

template<std::convertible_to<double> T>
[[nodiscard]] metrics_s<T> evaluate_metrics(const std::vector<T> &targets, const std::vector<T> &predictions) {
    return {.mae = mean_absolute_error(targets, predictions),
            .mse = mean_squared_error(targets, predictions),
            .rmse = root_mean_squared_error(targets, predictions),
            .r2_score = r2_score(targets, predictions)};
}

template<std::convertible_to<double> MetricsType>
struct first_and_last_metric_records_s {
    std::optional<metrics_s<MetricsType>> first_train_record;
    std::optional<metrics_s<MetricsType>> last_train_record;

    void print_diff() const noexcept {
        assert(first_train_record);
        assert(last_train_record);
        std::println("MAE [ before: {} | after: {} ]", first_train_record->mae, last_train_record->mae);
        std::println("MSE [ before: {} | after: {} ]", first_train_record->mse, last_train_record->mse);
        std::println("RMSE [ before: {} | after: {} ]", first_train_record->rmse, last_train_record->rmse);
        std::println("R2 [ before: {} | after: {} ]", first_train_record->r2_score, last_train_record->r2_score);
    }
};

} // namespace ml::metrics::regression