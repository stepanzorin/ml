// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "types.hpp"

namespace ml {

template<typename SampleT>
struct typed_dataset_s {
    std::vector<std::string> feature_names;
    std::vector<feature_info_s> features;
    target_info_s target_info;
    split_info_s split;
    std::vector<SampleT> samples;
    std::vector<SampleT> train_samples;
    std::vector<SampleT> test_samples;

    [[nodiscard]] std::size_t feature_count() const noexcept { return feature_names.size(); }

    [[nodiscard]] std::size_t row_count() const noexcept { return samples.size(); }

    [[nodiscard]] bool has_split() const noexcept { return split.has_split; }
};

using regression_dataset_s = typed_dataset_s<regression_sample_s>;
using binary_classification_dataset_s = typed_dataset_s<binary_classification_sample_s>;
using multiclass_classification_dataset_s = typed_dataset_s<multiclass_classification_sample_s>;
using multilabel_classification_dataset_s = typed_dataset_s<multilabel_classification_sample_s>;
using unsupervised_dataset_s = typed_dataset_s<unsupervised_sample_s>;

using dataset_variant_t = std::variant<regression_dataset_s,
                                       binary_classification_dataset_s,
                                       multiclass_classification_dataset_s,
                                       multilabel_classification_dataset_s,
                                       unsupervised_dataset_s>;

[[nodiscard]] std::string to_string(task_type_e task_type);
[[nodiscard]] std::string to_string(target_mode_e target_mode);

[[nodiscard]] tabular_dataset_s load_preprocessed_json_dataset(const std::filesystem::path &path);

[[nodiscard]] regression_dataset_s as_regression_dataset(const tabular_dataset_s &raw);
[[nodiscard]] binary_classification_dataset_s as_binary_classification_dataset(const tabular_dataset_s &raw);
[[nodiscard]] multiclass_classification_dataset_s as_multiclass_classification_dataset(const tabular_dataset_s &raw);
[[nodiscard]] multilabel_classification_dataset_s as_multilabel_classification_dataset(const tabular_dataset_s &raw);
[[nodiscard]] unsupervised_dataset_s as_unsupervised_dataset(const tabular_dataset_s &raw);

[[nodiscard]] regression_dataset_s load_regression_dataset(const std::filesystem::path &path);
[[nodiscard]] binary_classification_dataset_s load_binary_classification_dataset(const std::filesystem::path &path);
[[nodiscard]] multiclass_classification_dataset_s load_multiclass_classification_dataset(
        const std::filesystem::path &path);
[[nodiscard]] multilabel_classification_dataset_s load_multilabel_classification_dataset(
        const std::filesystem::path &path);
[[nodiscard]] unsupervised_dataset_s load_unsupervised_dataset(const std::filesystem::path &path);

[[nodiscard]] dataset_variant_t as_auto_dataset(const tabular_dataset_s &raw);
[[nodiscard]] dataset_variant_t load_auto_dataset(const std::filesystem::path &path);

} // namespace ml
