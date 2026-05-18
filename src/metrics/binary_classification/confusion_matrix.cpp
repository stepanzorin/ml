#include "confusion_matrix.hpp"

#include <cassert>
#include <ranges>

namespace ml::metrics::binary_classification {

double confusion_matrix_s::accuracy() const noexcept {
    const auto sum = true_positive + true_negative + false_positive + false_negative;

    if (sum == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positive + true_negative) / static_cast<double>(sum);
}
double confusion_matrix_s::precision() const noexcept {
    const auto denominator = true_positive + false_positive;

    if (denominator == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positive) / static_cast<double>(denominator);
}
double confusion_matrix_s::recall() const noexcept {
    const auto denominator = true_positive + false_negative;

    if (denominator == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positive) / static_cast<double>(denominator);
}
double confusion_matrix_s::f1_score() const noexcept {
    const auto precision_value = precision();
    const auto recall_value = recall();

    const auto denominator = precision_value + recall_value;

    if (denominator == 0.0) {
        return 0.0;
    }

    return 2.0 * precision_value * recall_value / denominator;
}

confusion_matrix_s make_confusion_matrix(const std::span<const std::uint32_t> targets,
                                         const std::span<const std::uint32_t> predictions) {
    assert(!targets.empty());
    assert(targets.size() == predictions.size());

    auto confusion_matrix = confusion_matrix_s{};

    for (const auto &&[target, prediction] : std::views::zip(targets, predictions)) {
        assert(target <= 1);
        assert(prediction <= 1);

        if (target == 1 && prediction == 1) {
            ++confusion_matrix.true_positive;
        } else if (target == 0 && prediction == 0) {
            ++confusion_matrix.true_negative;
        } else if (target == 0 && prediction == 1) {
            ++confusion_matrix.false_positive;
        } else if (target == 1 && prediction == 0) {
            ++confusion_matrix.false_negative;
        }
    }

    return confusion_matrix;
}

} // namespace ml::metrics::binary_classification