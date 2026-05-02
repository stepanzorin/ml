// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>

#include <cstdint>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <filesystem>
#include <vector>

#include "csv.hpp"
#include "util/detail/csv_parse_helpers.hpp"

namespace ml::util::detail {

struct file_closer {
    void operator()(FILE *const file) const noexcept {
        if (file) {
            std::fclose(file);
        }
    }
};

using unique_file_ptr_t = std::unique_ptr<FILE, file_closer>;


template<typename... ColumnTypes>
void validate_headers_count(const csv_headers_t &headers) {
    constexpr auto expected_column_count = sizeof...(ColumnTypes);

    if (headers.size() != expected_column_count) {
        throw std::runtime_error{
                std::format("Invalid header column count. Expected {}, got {}", expected_column_count, headers.size())};
    }
}

[[nodiscard]] char resolve_delimiter(std::optional<char> delimiter);

[[nodiscard]] bool is_space(char c) noexcept;

[[nodiscard]] bool is_empty_or_spaces(std::string_view line) noexcept;

[[nodiscard]] std::string trim(std::string_view value);

[[nodiscard]] std::uintmax_t get_file_size(const std::filesystem::path &path);

[[nodiscard]] unique_file_ptr_t open_file_for_reading(const std::filesystem::path &path);

[[nodiscard]] std::string read_file_content(const std::filesystem::path &path);

[[nodiscard]] std::vector<raw_csv_cell_s> split_csv_record(std::string_view record,
                                                           char delimiter,
                                                           std::size_t line_number);

[[nodiscard]] csv_headers_t make_headers(const std::vector<raw_csv_cell_s> &cells, std::size_t line_number);

[[nodiscard]] raw_csv_content_s parse_raw_csv(const std::filesystem::path &path, char delimiter, bool has_headers);

} // namespace ml::util::detail