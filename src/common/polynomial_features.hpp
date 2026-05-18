// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <vector>

#include "common/traits.hpp"

namespace ml::common {

namespace detail {

template<std::copy_constructible T>
[[nodiscard]] std::vector<T> concat_vectors(const std::vector<T> &a, const std::vector<T> &b) {
    if (a.empty()) {
        return b;
    }

    if (b.empty()) {
        return a;
    }

    auto result = std::vector<T>{};

    result.reserve(a.size() + b.size());

    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());

    return result;
}

template<feature T>
[[nodiscard]] std::vector<T> expand_with_powers(const std::vector<T> &features,
                                                const std::vector<std::uint32_t> &degrees) {
    assert(!features.empty());

    if (degrees.empty()) {
        return features;
    }

    const auto total_count = features.size() + (degrees.size() * features.size());

    auto features_with_powers = std::vector<T>{};
    features_with_powers.reserve(total_count);
    features_with_powers.insert(features_with_powers.end(), features.begin(), features.end());

    for (const auto degree : degrees) {
        auto powered = std::views::transform(features, [=](const T feature) { return std::pow(feature, degree); }) |
                       std::ranges::to<std::vector<T>>();
        features_with_powers.insert(features_with_powers.end(), powered.begin(), powered.end());
    }

    return features_with_powers;
}

template<feature T>
[[nodiscard]] std::vector<T> generate_interaction_features(const std::vector<T> &features_with_powers) {
    auto interaction_features = std::vector<T>{};

    for (const auto i : std::views::iota(0uz, features_with_powers.size())) {
        for (const auto j : std::views::iota(i + 1uz, features_with_powers.size())) {
            interaction_features.push_back(features_with_powers[i] * features_with_powers[j]);
        }
    }

    return interaction_features;
}

} // namespace detail

template<feature T>
[[nodiscard]] std::vector<T> generate_polynomial_features(const std::vector<T> &features,
                                                          const std::vector<std::uint32_t> &degrees) {
    const auto degree_expanded_features = detail::expand_with_powers(features, degrees);
    const auto interaction_features = detail::generate_interaction_features(features);
    return detail::concat_vectors(degree_expanded_features, interaction_features);
}

} // namespace ml::common