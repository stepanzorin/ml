// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstdint>
#include <vector>

namespace ml::metrics::binary_classification {

struct confusion_matrix_s {
    std::uint32_t true_positive = 0;
    std::uint32_t true_negative = 0;
    std::uint32_t false_positive = 0;
    std::uint32_t false_negative = 0;

    [[nodiscard]] double accuracy() const noexcept;
    [[nodiscard]] double precision() const noexcept;
    [[nodiscard]] double recall() const noexcept;
    [[nodiscard]] double f1_score() const noexcept;
};

[[nodiscard]] confusion_matrix_s make_confusion_matrix(const std::vector<std::uint32_t> &targets,
                                                       const std::vector<std::uint32_t> &predictions);

} // namespace ml::metrics::binary_classification