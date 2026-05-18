// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

#include "algorithms/euclidean_distance.hpp"

namespace ml::algorithms {

struct neighbor_s {
    std::size_t index;
    double squared_distance;
};

template<typename SampleType>
[[nodiscard]] std::vector<neighbor_s> find_k_nearest_neighbors(const std::size_t k,
                                                               const std::span<const SampleType> samples,
                                                               const std::span<const double> query_features) {
    assert(k != 0);
    assert(!samples.empty());
    assert(!query_features.empty());
    assert(k <= samples.size());

    const auto neighbor_less = [](const neighbor_s &a, const neighbor_s &b) {
        if (a.squared_distance != b.squared_distance) {
            return a.squared_distance < b.squared_distance;
        }

        return a.index < b.index;
    };

    auto neighbors = std::vector<neighbor_s>{};
    neighbors.reserve(k);

    for (const auto &[index, sample] : std::views::enumerate(samples)) {
        const auto candidate = neighbor_s{
                .index = static_cast<std::size_t>(index),
                .squared_distance = compute_squared_euclidean_distance(sample.features, query_features)};

        if (neighbors.size() < k) {
            neighbors.push_back(candidate);
            if (neighbors.size() == k) {
                std::ranges::make_heap(neighbors, neighbor_less);
            }
        } else if (neighbor_less(candidate, neighbors.front())) {
            std::ranges::pop_heap(neighbors, neighbor_less);
            neighbors.back() = candidate;
            std::ranges::push_heap(neighbors, neighbor_less);
        }
    }

    std::ranges::sort_heap(neighbors, neighbor_less);

    return neighbors;
}

} // namespace ml::algorithms