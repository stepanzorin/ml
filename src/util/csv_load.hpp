// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <filesystem>
#include <optional>

#include "util/detail/csv_load.hpp"
#include "util/detail/csv_parse_helpers.hpp"

namespace ml::util {

template<typename... ColumnTypes>
[[nodiscard]] typed_csv_table_with_headers_s<ColumnTypes...> read_csv(const std::filesystem::path &path,
                                                                      const csv_read_options_s &options = {}) {
    static_assert(sizeof...(ColumnTypes) > 0, "CSV table must contain at least one column type");

    const auto raw = detail::parse_raw_csv(path, options, true);

    detail::validate_headers_count<ColumnTypes...>(*raw.headers);

    auto table = typed_csv_table_with_headers_s<ColumnTypes...>{};

    table.headers = *raw.headers;
    table.rows.reserve(raw.rows.size());

    for (const auto &raw_row : raw.rows) {
        table.rows.push_back(detail::parse_required_row<ColumnTypes...>(raw_row));
    }

    return table;
}

template<typename... ColumnTypes>
[[nodiscard]] typed_csv_table_s<ColumnTypes...> read_csv_without_headers(const std::filesystem::path &path,
                                                                         const csv_read_options_s &options = {}) {
    static_assert(sizeof...(ColumnTypes) > 0, "CSV table must contain at least one column type");

    const auto raw = detail::parse_raw_csv(path, options, false);

    auto table = typed_csv_table_s<ColumnTypes...>{};

    table.rows.reserve(raw.rows.size());

    for (const auto &raw_row : raw.rows) {
        table.rows.push_back(detail::parse_required_row<ColumnTypes...>(raw_row));
    }

    return table;
}

template<typename... ColumnTypes>
[[nodiscard]] nullable_typed_csv_table_with_headers_s<ColumnTypes...> read_nullable_csv(
        const std::filesystem::path &path,
        const csv_read_options_s &options = {}) {
    static_assert(sizeof...(ColumnTypes) > 0, "CSV table must contain at least one column type");

    const auto raw = detail::parse_raw_csv(path, options, true);

    detail::validate_headers_count<ColumnTypes...>(*raw.headers);

    auto table = nullable_typed_csv_table_with_headers_s<ColumnTypes...>{};

    table.headers = *raw.headers;
    table.rows.reserve(raw.rows.size());

    for (const auto &raw_row : raw.rows) {
        table.rows.push_back(detail::parse_nullable_row<ColumnTypes...>(raw_row));
    }

    return table;
}

template<typename... ColumnTypes>
[[nodiscard]] nullable_typed_csv_table_s<ColumnTypes...> read_nullable_csv_without_headers(
        const std::filesystem::path &path,
        const csv_read_options_s &options = {}) {
    static_assert(sizeof...(ColumnTypes) > 0, "CSV table must contain at least one column type");

    const auto raw = detail::parse_raw_csv(path, options, false);

    auto table = nullable_typed_csv_table_s<ColumnTypes...>{};

    table.rows.reserve(raw.rows.size());

    for (const auto &raw_row : raw.rows) {
        table.rows.push_back(detail::parse_nullable_row<ColumnTypes...>(raw_row));
    }

    return table;
}

} // namespace ml::util