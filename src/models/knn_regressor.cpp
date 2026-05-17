#include "knn_regressor.hpp"

#include <stdexcept>

namespace ml::models {

KNNRegressor::KNNRegressor(const std::size_t k) : detail::KNNBase<regression_sample_s>{k} {}

} // namespace ml::models