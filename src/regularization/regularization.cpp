#include "regularization.hpp"

#include <stdexcept>

namespace ml::regularization {

namespace {

[[nodiscard]] double sign(const double weight) noexcept {
    if (weight > 1.0) {
        return 1.0;
    }

    if (weight < -1.0) {
        return -1.0;
    }

    return 0.0;
}

} // namespace

void validate_regularization(const regularization_s &regularization) {
    switch (regularization.type) {
        case regularization_type_e::none:
            if (regularization.l1_lambda != 0.0 || regularization.l2_lambda != 0.0) {
                throw std::runtime_error{"No regularization selected, but lambda values are > 0"};
            }
            break;

        case regularization_type_e::l1_lasso:
            if (regularization.l1_lambda <= 0.0) {
                throw std::runtime_error{"L1 regularization requires l1_lambda > 0"};
            }
            if (regularization.l2_lambda != 0.0) {
                throw std::runtime_error{"L1 regularization cannot use l2_lambda"};
            }
            break;

        case regularization_type_e::l2_ridge:
            if (regularization.l2_lambda <= 0.0) {
                throw std::runtime_error{"L2 regularization requires l2_lambda > 0"};
            }
            if (regularization.l1_lambda != 0.0) {
                throw std::runtime_error{"L2 regularization cannot use l1_lambda"};
            }
            break;

        case regularization_type_e::elastic_net:
            if (regularization.l1_lambda < 0.0 || regularization.l2_lambda < 0.0) {
                throw std::runtime_error{"Elastic Net lambdas must not be negative"};
            }
            if (regularization.l1_lambda <= 0.0 && regularization.l2_lambda <= 0.0) {
                throw std::runtime_error{"Elastic Net requires at least one lambda > 0"};
            }
            break;
    }
}

double regularization_gradient(const regularization_s &regularization, const double weight) noexcept {
    switch (regularization.type) {
        case regularization_type_e::l1_lasso: return regularization.l1_lambda * sign(weight);
        case regularization_type_e::l2_ridge: return regularization.l2_lambda * weight;
        case regularization_type_e::elastic_net:
            return regularization.l1_lambda * sign(weight) + regularization.l2_lambda * weight;
        case regularization_type_e::none: [[fallthrough]];
        default: return 0.0;
    }
}

} // namespace ml::regularization