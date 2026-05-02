// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace ml {

struct csv_range_s {
    std::size_t offset{};
    std::optional<std::size_t> count = std::nullopt;
};

struct csv_read_options_s {
    std::optional<char> delimiter = std::nullopt;

    csv_range_s columns{};
    csv_range_s rows{};
};


using csv_headers_t = std::vector<std::string>;

struct raw_csv_row_s {
    std::size_t line_number{};
    std::vector<std::string> cells;
};

struct raw_csv_content_s {
    std::optional<csv_headers_t> headers;
    std::vector<raw_csv_row_s> rows;
};

template<typename... ColumnTypes>
using csv_row_t = std::tuple<ColumnTypes...>;

template<typename... ColumnTypes>
struct typed_csv_table_s {
    std::vector<csv_row_t<ColumnTypes...>> rows;

    [[nodiscard]] std::size_t row_count() const noexcept { return rows.size(); }
};

template<typename... ColumnTypes>
struct typed_csv_table_with_headers_s {
    csv_headers_t headers;
    std::vector<csv_row_t<ColumnTypes...>> rows;

    [[nodiscard]] std::size_t row_count() const noexcept { return rows.size(); }

    [[nodiscard]] static consteval std::size_t column_count() noexcept { return sizeof...(ColumnTypes); }
};


template<typename... ColumnTypes>
using nullable_csv_row_t = std::tuple<std::optional<ColumnTypes>...>;

template<typename... ColumnTypes>
struct nullable_typed_csv_table_s {
    std::vector<nullable_csv_row_t<ColumnTypes...>> rows;

    [[nodiscard]] std::size_t row_count() const noexcept { return rows.size(); }
};

template<typename... ColumnTypes>
struct nullable_typed_csv_table_with_headers_s {
    csv_headers_t headers;
    std::vector<nullable_csv_row_t<ColumnTypes...>> rows;

    [[nodiscard]] std::size_t row_count() const noexcept { return rows.size(); }

    [[nodiscard]] static consteval std::size_t column_count() noexcept { return sizeof...(ColumnTypes); }
};

} // namespace ml