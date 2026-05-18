// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "metrics/metrics.hpp"
#include "models/detail/knn_base.hpp"
#include "types.hpp"

namespace ml::models {

class KNNClassifier final : public detail::KNNBase<binary_classification_sample_s> {
public:
    KNNClassifier() = delete;

    explicit KNNClassifier(std::size_t k);

    [[nodiscard]] double predict_probability(const std::vector<double> &features) const noexcept;
    [[nodiscard]] std::uint32_t predict_label(const std::vector<double> &features,
                                              double threshold = 0.5) const noexcept;

    [[nodiscard]] metrics::binary_classification_metrics_s evaluate(
            const std::vector<binary_classification_sample_s> &samples,
            double threshold = 0.5) const;
};

} // namespace ml::models
