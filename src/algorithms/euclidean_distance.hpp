// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <span>

namespace ml::algorithms {

[[nodiscard]] double compute_squared_euclidean_distance(std::span<const double> lhs, std::span<const double> rhs);

} // namespace ml::algorithms