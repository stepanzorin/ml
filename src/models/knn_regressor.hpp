// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <vector>

#include "models/detail/knn_base.hpp"
#include "types.hpp"

namespace ml::models {

class KNNRegressor final : detail::KNNBase<regression_sample_s> {
public:
    KNNRegressor() = delete;

    explicit KNNRegressor(std::size_t k);
};

} // namespace ml::models
