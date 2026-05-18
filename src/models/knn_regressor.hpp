// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <vector>

#include "metrics/metrics.hpp"
#include "models/detail/knn_base.hpp"
#include "types.hpp"

namespace ml::models {

class KNNRegressor final : public detail::KNNBase<regression_sample_s> {
public:
    KNNRegressor() = delete;

    explicit KNNRegressor(std::size_t k);

    [[nodiscard]] double predict(const std::vector<double> &features) const noexcept;

    [[nodiscard]] metrics::regression_metrics_s evaluate(const std::vector<regression_sample_s> &samples) const;
};

} // namespace ml::models
