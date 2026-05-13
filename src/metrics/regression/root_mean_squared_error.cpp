#include "root_mean_squared_error.hpp"

#include <cmath>

#include "metrics/regression/mean_squared_error.hpp"

namespace ml::metrics::regression {

double root_mean_squared_error(const std::span<const double> targets, const std::span<const double> predictions) {
    return std::sqrt(mean_squared_error(targets, predictions));
}

} // namespace ml::metrics::regression