// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstdint>
#include <span>

namespace ml::metrics::binary_classification {

[[nodiscard]] double evaluate_log_loss(std::span<const std::uint32_t> targets, std::span<const double> probabilities);

} // namespace ml::metrics::binary_classification