// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <span>

namespace ml::metrics::regression {

[[nodiscard]] double mean_absolute_error(std::span<const double> targets, std::span<const double> predictions);

} // namespace ml::metrics::regression