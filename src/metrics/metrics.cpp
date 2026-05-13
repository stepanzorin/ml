#include "metrics.hpp"

#include <cassert>
#include <print>

#include "metrics/binary_classification/confusion_matrix.hpp"
#include "metrics/binary_classification/log_loss.hpp"
#include "metrics/regression/mean_absolute_error.hpp"
#include "metrics/regression/mean_squared_error.hpp"
#include "metrics/regression/r2_score.hpp"
#include "metrics/regression/root_mean_squared_error.hpp"

namespace ml::metrics {

regression_metrics_s evaluate_regression_metrics(const std::span<const double> targets,
                                                 const std::span<const double> predictions) {
    return {.mae = regression::mean_absolute_error(targets, predictions),
            .mse = regression::mean_squared_error(targets, predictions),
            .rmse = regression::root_mean_squared_error(targets, predictions),
            .r2_score = regression::r2_score(targets, predictions)};
}

void regression_metric_records_s::print_diff() const noexcept {
    assert(first);
    assert(last);
    std::println("MAE [ before: {} | after: {} ]", first->mae, last->mae);
    std::println("MSE [ before: {} | after: {} ]", first->mse, last->mse);
    std::println("RMSE [ before: {} | after: {} ]", first->rmse, last->rmse);
    std::println("R2 [ before: {} | after: {} ]", first->r2_score, last->r2_score);
}

binary_classification_metrics_s evaluate_binary_classification_metrics(const std::span<const std::uint32_t> targets,
                                                                       const std::span<const std::uint32_t> predictions,
                                                                       const std::span<const double> probabilities) {
    const auto confusion_matrix = binary_classification::make_confusion_matrix(targets, predictions);
    return {.accuracy = confusion_matrix.accuracy(),
            .precision = confusion_matrix.precision(),
            .recall = confusion_matrix.recall(),
            .f1_score = confusion_matrix.f1_score(),
            .log_loss = binary_classification::evaluate_log_loss(targets, probabilities)};
}

void binary_classification_metric_records_s::print_diff() const noexcept {
    assert(first);
    assert(last);
    std::println("Accuracy [ before: {} | after: {} ]", first->accuracy, last->accuracy);
    std::println("Precision [ before: {} | after: {} ]", first->precision, last->precision);
    std::println("Recall [ before: {} | after: {} ]", first->recall, last->recall);
    std::println("F1 Score [ before: {} | after: {} ]", first->f1_score, last->f1_score);
    std::println("LogLoss [ before: {} | after: {} ]", first->log_loss, last->log_loss);
}
} // namespace ml::metrics