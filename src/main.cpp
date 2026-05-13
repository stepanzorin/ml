#include <stdexcept>

#include <spdlog/spdlog.h>

#include "dataset_loader.hpp"
#include "models/linear_regression.hpp"
#include "util/filesystem_helpers.hpp"

int main() try {
    const auto dataset_path = ml::util::datasets_directory_path() / "ecommerce_sales_analytics_5000.json";
    const auto dataset = ml::load_regression_dataset(dataset_path);

    auto model = ml::models::LinearRegression(dataset.feature_count());

    model.train(dataset.train_samples,
                1'000,
                0.0001,
                ml::regularization::regularization_s{.type = ml::regularization::regularization_type_e::l2_ridge,
                                                     .l2_lambda = 0.001});

    model.print_parameters();

    return EXIT_SUCCESS;
} catch (const std::exception &ex) {
    spdlog::error(std::format("Unhandled exception caught: <{}>", ex.what()));
    return EXIT_FAILURE;
} catch (...) {
    spdlog::error("Unhandled exception caught: <Unknown error>");
    return EXIT_FAILURE;
}