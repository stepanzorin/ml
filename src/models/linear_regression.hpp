#pragma once

#include <print>
#include <vector>

#include "common/types.hpp"

namespace ml::models {

class LinearRegression {
public:
    explicit LinearRegression(const std::size_t feature_count) : m_bias{0.0}, m_weights(feature_count, 0.0) {}

    [[nodiscard]] double predict(const std::vector<double> &features);

    void train(const std::vector<common::sample_s<double>> &samples, std::uint32_t epoch_count, double learning_rate);

    void print_parameters() const noexcept;

private:
    double m_bias;
    std::vector<double> m_weights;
};

} // namespace ml::models