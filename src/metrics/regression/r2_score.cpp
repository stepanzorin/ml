#include "r2_score.hpp"

#include <algorithm>
#include <cassert>
#include <ranges>
#include <utility>

namespace ml::metrics::regression {

double r2_score(const std::span<const double> targets, const std::span<const double> predictions) {
    assert(!targets.empty());
    assert(targets.size() == predictions.size());

    const auto target_mean = std::ranges::fold_left(targets, 0.0, std::plus{}) / targets.size();

    auto ss_res = 0.0;
    auto ss_tot = 0.0;

    for (const auto &&[target, prediction] : std::views::zip(targets, predictions)) {
        ss_res += (target - prediction) * (target - prediction);
        ss_tot += (target - target_mean) * (target - target_mean);
    }

    assert(ss_tot != 0.0);

    return 1.0 - ss_res / ss_tot;
}

} // namespace ml::metrics::regression