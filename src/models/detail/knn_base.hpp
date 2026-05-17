// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "models/detail/knn_base.hpp"
#include "types.hpp"

namespace ml::models::detail {

struct neighbor_s {
    std::size_t sample_index{};
    double squared_distance{};
};

template<typename SampleType>
class KNNBase {
public:
    KNNBase() = delete;

    explicit KNNBase(const std::size_t k) : m_k{k} {}

    void fit(std::vector<SampleType> samples) {
        if (samples.empty()) {
            throw std::runtime_error{"Samples must not be empty"};
        }
        m_samples = std::move(samples);
        m_feature_count = m_samples.size();
    }

    virtual ~KNNBase() = default;

private:
    std::size_t m_k = 0;
    std::size_t m_feature_count = 0;
    std::vector<SampleType> m_samples = {};
};

} // namespace ml::models::detail
