// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

namespace ml::regularization {

enum class regularization_type_e { none, l1_lasso, l2_ridge, elastic_net };

struct regularization_s {
    regularization_type_e type;

    double l1_lambda = 0.0;
    double l2_lambda = 0.0;
};

void validate_regularization(const regularization_s &regularization);

[[nodiscard]] double regularization_gradient(const regularization_s &regularization, double weight) noexcept;

} // namespace ml::regularization