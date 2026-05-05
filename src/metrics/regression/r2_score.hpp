// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace ml::metrics::regression {

template<std::convertible_to<double> T>
[[nodiscard]] double r2_score(const std::vector<T> &targets, const std::vector<T> &predictions) {
    if (targets.size() != predictions.size()) {
        throw std::runtime_error{"Targets and predictions size mismatch"};
    }

    if (targets.empty()) {
        throw std::runtime_error{"Targets must not be empty"};
    }

    const auto target_mean = std::ranges::fold_left(targets, 0.0, std::plus{}) / targets.size();

    auto ss_res = 0.0;
    auto ss_tot = 0.0;

    for (const auto &&[target, prediction] : std::views::zip(targets, predictions)) {
        ss_res += std::pow(target - prediction, 2);
        ss_tot += std::pow(target - target_mean, 2);
    }

    if (ss_tot == 0.0) {
        throw std::runtime_error{"R2 score is undefined when all targets are equal"};
    }

    return 1.0 - ss_res / ss_tot;
}

} // namespace ml::metrics::regression