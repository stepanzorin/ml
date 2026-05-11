// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ml {

enum class task_type_e {
    regression,
    binary_classification,
    multiclass_classification,
    multilabel_classification,
    unsupervised
};

enum class target_mode_e { none, numeric, class_id, multilabel };

struct feature_info_s {
    std::string name;
    std::string source_column;
    std::string kind;
};

struct split_info_s {
    bool has_split{false};
    double train_ratio{};
    bool shuffle{};
    int seed{};
    std::size_t train_row_count{};
    std::size_t test_row_count{};
};

struct target_info_s {
    target_mode_e mode{target_mode_e::none};
    std::string column;
    std::vector<std::string> columns;
    std::unordered_map<std::string, std::uint32_t> class_to_id;
    std::vector<std::string> id_to_class;
};

class target_value_s {
public:
    using payload_t = std::variant<std::monostate, double, std::uint32_t, std::vector<double>>;

    target_value_s() = default;

    [[nodiscard]] static target_value_s none();
    [[nodiscard]] static target_value_s numeric(double value);
    [[nodiscard]] static target_value_s class_id(std::uint32_t value);
    [[nodiscard]] static target_value_s multilabel(std::vector<double> value);

    [[nodiscard]] target_mode_e mode() const noexcept;
    [[nodiscard]] bool is_none() const noexcept;
    [[nodiscard]] double numeric() const;
    [[nodiscard]] std::uint32_t class_id() const;
    [[nodiscard]] const std::vector<double> &multilabel() const;

private:
    target_mode_e mode_{target_mode_e::none};
    payload_t value_{};
};

struct tabular_sample_s {
    std::vector<double> features;
    target_value_s target;
};

struct tabular_dataset_s {
    task_type_e task_type{task_type_e::regression};
    target_info_s target_info;
    std::vector<std::string> feature_names;
    std::vector<feature_info_s> features;
    std::vector<tabular_sample_s> samples;
    std::vector<tabular_sample_s> train_samples;
    std::vector<tabular_sample_s> test_samples;
    split_info_s split;

    [[nodiscard]] std::size_t feature_count() const noexcept;
    [[nodiscard]] std::size_t row_count() const noexcept;
    [[nodiscard]] bool has_split() const noexcept;
};

struct regression_sample_s {
    std::vector<double> features;
    double target{};
};

struct binary_classification_sample_s {
    std::vector<double> features;
    std::uint32_t label{};
};

struct multiclass_classification_sample_s {
    std::vector<double> features;
    std::uint32_t class_id{};
};

struct multilabel_classification_sample_s {
    std::vector<double> features;
    std::vector<double> labels;
};

struct unsupervised_sample_s {
    std::vector<double> features;
};

} // namespace ml