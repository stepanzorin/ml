// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <vector>

namespace ml::models::detail {

class LinearScoringModelBase {
public:
    explicit LinearScoringModelBase(const std::size_t feature_count) : m_bias{0.0}, m_weights(feature_count, 0.0) {}

    virtual ~LinearScoringModelBase() = default;

protected:
    double m_bias;
    std::vector<double> m_weights;

    // score = bias + w0*x0 + w1*x1 + ...
    [[nodiscard]] double score(const std::vector<double> &features) const;

    void print_basic_parameters() const noexcept;
};

} // namespace ml::models::detail
