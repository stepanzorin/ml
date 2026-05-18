// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

namespace ml::models::detail {

struct neighbor_s {
    std::size_t sample_index{};
    double squared_distance{};
};

template<typename SampleType>
class KNNBase {
public:
    KNNBase() = delete;

    explicit KNNBase(const std::size_t k) : m_k{k} { assert(k != 0); }

    void fit(const std::vector<SampleType> &samples) {
        assert(!samples.empty());
        assert(samples.size() >= m_k);

        m_sample_feature_count = samples.front().features.size();
        assert(m_sample_feature_count > 0);

        m_samples = samples;
    }

    virtual ~KNNBase() = default;

protected:
    std::size_t m_k = 0;
    std::size_t m_sample_feature_count = 0;
    std::vector<SampleType> m_samples = {};
};

} // namespace ml::models::detail
