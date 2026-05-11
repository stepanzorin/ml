#include <print>

#include "dataset_loader.hpp"
#include "util/filesystem_helpers.hpp"

int main() {
    const auto dataset_path = ml::util::datasets_directory_path() / "ecommerce_sales_analytics_5000.json";
    const auto dataset = ml::load_regression_dataset(dataset_path);

    std::println("Rows: {}", dataset.row_count());
    std::println("Features: {}", dataset.feature_count());

    return 0;
}