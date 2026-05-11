#include "dataset_loader.hpp"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

#include <rapidjson/document.h>

namespace ml {

namespace {

namespace detail {

[[nodiscard]] std::string read_text_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error{"Cannot open JSON file: " + path.string()};
    }

    return std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

[[nodiscard]] const rapidjson::Value &require_member(const rapidjson::Value &object, const char *name) {
    if (!object.IsObject()) {
        throw std::runtime_error{"Expected JSON object"};
    }

    const auto it = object.FindMember(name);
    if (it == object.MemberEnd()) {
        throw std::runtime_error{std::string{"Missing JSON member: "} + name};
    }

    return it->value;
}

[[nodiscard]] const rapidjson::Value *find_member(const rapidjson::Value &object, const char *name) {
    if (!object.IsObject()) {
        return nullptr;
    }

    const auto it = object.FindMember(name);
    if (it == object.MemberEnd()) {
        return nullptr;
    }

    return &it->value;
}

[[nodiscard]] std::string require_string(const rapidjson::Value &object, const char *name) {
    const auto &value = require_member(object, name);
    if (!value.IsString()) {
        throw std::runtime_error{std::string{"Expected string member: "} + name};
    }

    return value.GetString();
}

[[nodiscard]] double require_number(const rapidjson::Value &object, const char *name) {
    const auto &value = require_member(object, name);
    if (!value.IsNumber()) {
        throw std::runtime_error{std::string{"Expected numeric member: "} + name};
    }

    return value.GetDouble();
}

[[nodiscard]] bool require_bool(const rapidjson::Value &object, const char *name) {
    const auto &value = require_member(object, name);
    if (!value.IsBool()) {
        throw std::runtime_error{std::string{"Expected bool member: "} + name};
    }

    return value.GetBool();
}

[[nodiscard]] int require_int(const rapidjson::Value &object, const char *name) {
    const auto &value = require_member(object, name);
    if (!value.IsInt()) {
        throw std::runtime_error{std::string{"Expected int member: "} + name};
    }

    return value.GetInt();
}

[[nodiscard]] std::size_t require_size(const rapidjson::Value &object, const char *name) {
    const auto value = require_number(object, name);
    if (value < 0.0) {
        throw std::runtime_error{std::string{"Expected non-negative size member: "} + name};
    }

    return static_cast<std::size_t>(value);
}

[[nodiscard]] task_type_e parse_task_type(const std::string &value) {
    if (value == "regression") {
        return task_type_e::regression;
    }
    if (value == "binary_classification") {
        return task_type_e::binary_classification;
    }
    if (value == "multiclass_classification") {
        return task_type_e::multiclass_classification;
    }
    if (value == "multilabel_classification") {
        return task_type_e::multilabel_classification;
    }
    if (value == "unsupervised") {
        return task_type_e::unsupervised;
    }

    throw std::runtime_error{"Unknown task_type: " + value};
}

[[nodiscard]] target_mode_e parse_target_mode(const std::string &value) {
    if (value == "none") {
        return target_mode_e::none;
    }
    if (value == "numeric") {
        return target_mode_e::numeric;
    }
    if (value == "class_id") {
        return target_mode_e::class_id;
    }
    if (value == "multilabel") {
        return target_mode_e::multilabel;
    }

    throw std::runtime_error{"Unknown target mode: " + value};
}

[[nodiscard]] std::vector<std::string> read_string_array(const rapidjson::Value &array, const char *name) {
    if (!array.IsArray()) {
        throw std::runtime_error{std::string{name} + " must be an array"};
    }

    std::vector<std::string> result;
    result.reserve(array.Size());

    for (const auto &value : array.GetArray()) {
        if (!value.IsString()) {
            throw std::runtime_error{std::string{name} + " must contain only strings"};
        }

        result.emplace_back(value.GetString());
    }

    return result;
}

[[nodiscard]] std::vector<double> read_number_array(const rapidjson::Value &array, const char *name) {
    if (!array.IsArray()) {
        throw std::runtime_error{std::string{name} + " must be an array"};
    }

    std::vector<double> result;
    result.reserve(array.Size());

    for (const auto &value : array.GetArray()) {
        if (!value.IsNumber()) {
            throw std::runtime_error{std::string{name} + " must contain only numbers"};
        }

        result.push_back(value.GetDouble());
    }

    return result;
}

[[nodiscard]] target_info_s read_target_info(const rapidjson::Value &object) {
    target_info_s info;
    info.mode = parse_target_mode(require_string(object, "mode"));

    if (const auto *column = find_member(object, "column")) {
        if (!column->IsString()) {
            throw std::runtime_error{"target.column must be a string"};
        }
        info.column = column->GetString();
    }

    if (const auto *columns = find_member(object, "columns")) {
        info.columns = read_string_array(*columns, "target.columns");
    }

    if (const auto *id_to_class = find_member(object, "id_to_class")) {
        info.id_to_class = read_string_array(*id_to_class, "target.id_to_class");
    }

    if (const auto *class_to_id = find_member(object, "class_to_id")) {
        if (!class_to_id->IsObject()) {
            throw std::runtime_error{"target.class_to_id must be an object"};
        }

        for (auto it = class_to_id->MemberBegin(); it != class_to_id->MemberEnd(); ++it) {
            if (!it->name.IsString() || !it->value.IsUint()) {
                throw std::runtime_error{"target.class_to_id must map strings to unsigned integers"};
            }

            info.class_to_id.emplace(it->name.GetString(), it->value.GetUint());
        }
    }

    return info;
}

[[nodiscard]] target_value_s read_target_value(const rapidjson::Value &sample_object, const target_mode_e mode) {
    if (mode == target_mode_e::none) {
        return target_value_s::none();
    }

    const auto &target_object = require_member(sample_object, "target");
    if (!target_object.IsObject()) {
        throw std::runtime_error{"sample.target must be an object"};
    }

    switch (mode) {
        case target_mode_e::numeric: return target_value_s::numeric(require_number(target_object, "numeric"));

        case target_mode_e::class_id: {
            const auto &class_id = require_member(target_object, "class_id");
            if (!class_id.IsUint()) {
                throw std::runtime_error{"sample.target.class_id must be unsigned integer"};
            }
            return target_value_s::class_id(static_cast<std::uint32_t>(class_id.GetUint()));
        }

        case target_mode_e::multilabel:
            return target_value_s::multilabel(
                    read_number_array(require_member(target_object, "labels"), "sample.target.labels"));

        case target_mode_e::none: return target_value_s::none();
    }

    throw std::runtime_error{"Unsupported target mode"};
}

[[nodiscard]] std::vector<tabular_sample_s> read_samples_array(const rapidjson::Value &samples,
                                                               const std::size_t expected_feature_count,
                                                               const target_mode_e target_mode) {
    if (!samples.IsArray()) {
        throw std::runtime_error{"samples value must be an array"};
    }

    std::vector<tabular_sample_s> result;
    result.reserve(samples.Size());

    for (const auto &sample_value : samples.GetArray()) {
        if (!sample_value.IsObject()) {
            throw std::runtime_error{"Each sample must be an object"};
        }

        tabular_sample_s sample;
        sample.features = read_number_array(require_member(sample_value, "features"), "sample.features");
        if (sample.features.size() != expected_feature_count) {
            throw std::runtime_error{"Sample feature count does not match feature_names count"};
        }

        sample.target = read_target_value(sample_value, target_mode);
        result.push_back(std::move(sample));
    }

    return result;
}

[[nodiscard]] split_info_s read_split_info(const rapidjson::Value &root) {
    split_info_s split;

    const auto *split_value = find_member(root, "split");
    if (!split_value) {
        return split;
    }

    if (!split_value->IsObject()) {
        throw std::runtime_error{"split must be an object"};
    }

    if (const auto *has_split = find_member(*split_value, "has_split")) {
        if (!has_split->IsBool()) {
            throw std::runtime_error{"split.has_split must be bool"};
        }
        split.has_split = has_split->GetBool();
    } else {
        split.has_split = true;
    }

    if (!split.has_split) {
        return split;
    }

    split.train_ratio = require_number(*split_value, "train_ratio");
    split.shuffle = require_bool(*split_value, "shuffle");
    split.seed = require_int(*split_value, "seed");
    split.train_row_count = require_size(*split_value, "train_row_count");
    split.test_row_count = require_size(*split_value, "test_row_count");

    return split;
}

template<typename SampleT, typename Converter>
[[nodiscard]] typed_dataset_s<SampleT> convert_dataset(const tabular_dataset_s &raw, Converter &&converter) {
    typed_dataset_s<SampleT> result;
    result.feature_names = raw.feature_names;
    result.features = raw.features;
    result.target_info = raw.target_info;
    result.split = raw.split;

    result.samples.reserve(raw.samples.size());
    for (const auto &sample : raw.samples) {
        result.samples.push_back(converter(sample));
    }

    result.train_samples.reserve(raw.train_samples.size());
    for (const auto &sample : raw.train_samples) {
        result.train_samples.push_back(converter(sample));
    }

    result.test_samples.reserve(raw.test_samples.size());
    for (const auto &sample : raw.test_samples) {
        result.test_samples.push_back(converter(sample));
    }

    return result;
}

} // namespace detail

} // namespace

target_value_s target_value_s::none() {
    target_value_s target;
    target.mode_ = target_mode_e::none;
    target.value_ = std::monostate{};
    return target;
}

target_value_s target_value_s::numeric(const double value) {
    target_value_s target;
    target.mode_ = target_mode_e::numeric;
    target.value_ = value;
    return target;
}

target_value_s target_value_s::class_id(const std::uint32_t value) {
    target_value_s target;
    target.mode_ = target_mode_e::class_id;
    target.value_ = value;
    return target;
}

target_value_s target_value_s::multilabel(std::vector<double> value) {
    target_value_s target;
    target.mode_ = target_mode_e::multilabel;
    target.value_ = std::move(value);
    return target;
}

target_mode_e target_value_s::mode() const noexcept { return mode_; }

bool target_value_s::is_none() const noexcept { return mode_ == target_mode_e::none; }

double target_value_s::numeric() const {
    if (mode_ != target_mode_e::numeric) {
        throw std::runtime_error{"Target is not numeric"};
    }

    return std::get<double>(value_);
}

std::uint32_t target_value_s::class_id() const {
    if (mode_ != target_mode_e::class_id) {
        throw std::runtime_error{"Target is not class_id"};
    }

    return std::get<std::uint32_t>(value_);
}

const std::vector<double> &target_value_s::multilabel() const {
    if (mode_ != target_mode_e::multilabel) {
        throw std::runtime_error{"Target is not multilabel"};
    }

    return std::get<std::vector<double>>(value_);
}

std::size_t tabular_dataset_s::feature_count() const noexcept { return feature_names.size(); }

std::size_t tabular_dataset_s::row_count() const noexcept { return samples.size(); }

bool tabular_dataset_s::has_split() const noexcept { return split.has_split; }

std::string to_string(const task_type_e task_type) {
    switch (task_type) {
        case task_type_e::regression: return "regression";
        case task_type_e::binary_classification: return "binary_classification";
        case task_type_e::multiclass_classification: return "multiclass_classification";
        case task_type_e::multilabel_classification: return "multilabel_classification";
        case task_type_e::unsupervised: return "unsupervised";
    }

    throw std::runtime_error{"Unknown task_type enum value"};
}

std::string to_string(const target_mode_e target_mode) {
    switch (target_mode) {
        case target_mode_e::none: return "none";
        case target_mode_e::numeric: return "numeric";
        case target_mode_e::class_id: return "class_id";
        case target_mode_e::multilabel: return "multilabel";
    }

    throw std::runtime_error{"Unknown target_mode enum value"};
}

tabular_dataset_s load_preprocessed_json_dataset(const std::filesystem::path &path) {
    const auto json_text = detail::read_text_file(path);

    rapidjson::Document document;
    document.Parse(json_text.c_str());

    if (document.HasParseError()) {
        throw std::runtime_error{"Invalid JSON: parse error"};
    }
    if (!document.IsObject()) {
        throw std::runtime_error{"Root JSON value must be an object"};
    }

    tabular_dataset_s dataset;
    dataset.task_type = detail::parse_task_type(detail::require_string(document, "task_type"));
    dataset.target_info = detail::read_target_info(detail::require_member(document, "target"));

    dataset.feature_names = detail::read_string_array(detail::require_member(document, "feature_names"),
                                                      "feature_names");

    const auto &features = detail::require_member(document, "features");
    if (!features.IsArray()) {
        throw std::runtime_error{"features must be an array"};
    }

    dataset.features.reserve(features.Size());
    for (const auto &value : features.GetArray()) {
        if (!value.IsObject()) {
            throw std::runtime_error{"features entries must be objects"};
        }

        feature_info_s info;
        info.name = detail::require_string(value, "name");
        info.source_column = detail::require_string(value, "source_column");
        info.kind = detail::require_string(value, "kind");
        dataset.features.push_back(std::move(info));
    }

    if (dataset.features.size() != dataset.feature_names.size()) {
        throw std::runtime_error{"features count does not match feature_names count"};
    }

    dataset.split = detail::read_split_info(document);
    dataset.samples = detail::read_samples_array(detail::require_member(document, "samples"),
                                                 dataset.feature_names.size(),
                                                 dataset.target_info.mode);

    if (dataset.split.has_split) {
        dataset.train_samples = detail::read_samples_array(detail::require_member(document, "train_samples"),
                                                           dataset.feature_names.size(),
                                                           dataset.target_info.mode);
        dataset.test_samples = detail::read_samples_array(detail::require_member(document, "test_samples"),
                                                          dataset.feature_names.size(),
                                                          dataset.target_info.mode);

        if (dataset.train_samples.size() != dataset.split.train_row_count) {
            throw std::runtime_error{"train_samples size does not match split.train_row_count"};
        }
        if (dataset.test_samples.size() != dataset.split.test_row_count) {
            throw std::runtime_error{"test_samples size does not match split.test_row_count"};
        }
    }

    return dataset;
}

regression_dataset_s as_regression_dataset(const tabular_dataset_s &raw) {
    if (raw.task_type != task_type_e::regression || raw.target_info.mode != target_mode_e::numeric) {
        throw std::runtime_error{"Dataset is not a regression dataset"};
    }

    return detail::convert_dataset<regression_sample_s>(raw, [](const tabular_sample_s &sample) {
        return regression_sample_s{sample.features, sample.target.numeric()};
    });
}

binary_classification_dataset_s as_binary_classification_dataset(const tabular_dataset_s &raw) {
    if (raw.task_type != task_type_e::binary_classification || raw.target_info.mode != target_mode_e::class_id) {
        throw std::runtime_error{"Dataset is not a binary classification dataset"};
    }

    return detail::convert_dataset<binary_classification_sample_s>(raw, [](const tabular_sample_s &sample) {
        const auto label = sample.target.class_id();
        if (label > 1u) {
            throw std::runtime_error{"Binary classification label must be 0 or 1"};
        }
        return binary_classification_sample_s{sample.features, label};
    });
}

multiclass_classification_dataset_s as_multiclass_classification_dataset(const tabular_dataset_s &raw) {
    if (raw.task_type != task_type_e::multiclass_classification || raw.target_info.mode != target_mode_e::class_id) {
        throw std::runtime_error{"Dataset is not a multiclass classification dataset"};
    }

    return detail::convert_dataset<multiclass_classification_sample_s>(raw, [](const tabular_sample_s &sample) {
        return multiclass_classification_sample_s{sample.features, sample.target.class_id()};
    });
}

multilabel_classification_dataset_s as_multilabel_classification_dataset(const tabular_dataset_s &raw) {
    if (raw.task_type != task_type_e::multilabel_classification || raw.target_info.mode != target_mode_e::multilabel) {
        throw std::runtime_error{"Dataset is not a multilabel classification dataset"};
    }

    return detail::convert_dataset<multilabel_classification_sample_s>(raw, [](const tabular_sample_s &sample) {
        return multilabel_classification_sample_s{sample.features, sample.target.multilabel()};
    });
}

unsupervised_dataset_s as_unsupervised_dataset(const tabular_dataset_s &raw) {
    if (raw.task_type != task_type_e::unsupervised || raw.target_info.mode != target_mode_e::none) {
        throw std::runtime_error{"Dataset is not an unsupervised dataset"};
    }

    return detail::convert_dataset<unsupervised_sample_s>(raw, [](const tabular_sample_s &sample) {
        return unsupervised_sample_s{sample.features};
    });
}

regression_dataset_s load_regression_dataset(const std::filesystem::path &path) {
    return as_regression_dataset(load_preprocessed_json_dataset(path));
}

binary_classification_dataset_s load_binary_classification_dataset(const std::filesystem::path &path) {
    return as_binary_classification_dataset(load_preprocessed_json_dataset(path));
}

multiclass_classification_dataset_s load_multiclass_classification_dataset(const std::filesystem::path &path) {
    return as_multiclass_classification_dataset(load_preprocessed_json_dataset(path));
}

multilabel_classification_dataset_s load_multilabel_classification_dataset(const std::filesystem::path &path) {
    return as_multilabel_classification_dataset(load_preprocessed_json_dataset(path));
}

unsupervised_dataset_s load_unsupervised_dataset(const std::filesystem::path &path) {
    return as_unsupervised_dataset(load_preprocessed_json_dataset(path));
}

dataset_variant_t as_auto_dataset(const tabular_dataset_s &raw) {
    switch (raw.task_type) {
        case task_type_e::regression: return as_regression_dataset(raw);
        case task_type_e::binary_classification: return as_binary_classification_dataset(raw);
        case task_type_e::multiclass_classification: return as_multiclass_classification_dataset(raw);
        case task_type_e::multilabel_classification: return as_multilabel_classification_dataset(raw);
        case task_type_e::unsupervised: return as_unsupervised_dataset(raw);
    }

    throw std::runtime_error{"Unsupported task type"};
}

dataset_variant_t load_auto_dataset(const std::filesystem::path &path) {
    return as_auto_dataset(load_preprocessed_json_dataset(path));
}


} // namespace ml
