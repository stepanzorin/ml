// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace ml {

using csv_headers_t = std::vector<std::string>;


struct raw_csv_cell_s {
    std::string value;
    bool quoted = false;
};

struct raw_csv_row_s {
    std::size_t line_number;
    std::vector<raw_csv_cell_s> cells;
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