// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

#include <charconv>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace ml::util::detail {

template<typename>
inline constexpr bool always_false_v = false;

template<typename T>
inline constexpr bool is_year_month_day_v = std::same_as<std::remove_cvref_t<T>, std::chrono::year_month_day>;

template<typename T>
inline constexpr bool is_sys_days_v = std::same_as<std::remove_cvref_t<T>, std::chrono::sys_days>;


template<typename T>
[[nodiscard]] T parse_integer_cell(const std::string &value,
                                   const std::size_t line_number,
                                   const std::size_t column_number) {
    auto result = T{};

    const auto *begin = value.data();
    const auto *end = value.data() + value.size();

    const auto [ptr, error] = std::from_chars(begin, end, result);

    if (error == std::errc::invalid_argument) {
        throw std::runtime_error{
                std::format("Invalid integer value '{}' at line {}, column {}", value, line_number, column_number)};
    }

    if (error == std::errc::result_out_of_range) {
        throw std::runtime_error{std::format("Integer value out of range '{}' at line {}, column {}",
                                             value,
                                             line_number,
                                             column_number)};
    }

    if (ptr != end) {
        throw std::runtime_error{std::format("Invalid trailing characters in integer value '{}' at line {}, column {}",
                                             value,
                                             line_number,
                                             column_number)};
    }

    return result;
}

template<typename T>
[[nodiscard]] T parse_floating_point_cell(const std::string &value,
                                          const std::size_t line_number,
                                          const std::size_t column_number) {
    errno = 0;

    char *end = nullptr;
    const auto parsed = std::strtold(value.c_str(), &end);

    if (end == value.c_str()) {
        throw std::runtime_error{std::format("Invalid floating point value '{}' at line {}, column {}",
                                             value,
                                             line_number,
                                             column_number)};
    }

    if (*end != '\0') {
        throw std::runtime_error{
                std::format("Invalid trailing characters in floating point value '{}' at line {}, column {}",
                            value,
                            line_number,
                            column_number)};
    }

    if (errno == ERANGE) {
        throw std::runtime_error{std::format("Floating point value out of range '{}' at line {}, column {}",
                                             value,
                                             line_number,
                                             column_number)};
    }

    if (parsed > static_cast<long double>(std::numeric_limits<T>::max()) ||
        parsed < static_cast<long double>(std::numeric_limits<T>::lowest())) {
        throw std::runtime_error{std::format("Floating point value out of target type range '{}' at line {}, column {}",
                                             value,
                                             line_number,
                                             column_number)};
    }

    return static_cast<T>(parsed);
}

[[nodiscard]] inline bool parse_bool_cell(const std::string &value,
                                          const std::size_t line_number,
                                          const std::size_t column_number) {
    if (value == "1" || value == "true" || value == "TRUE" || value == "True") {
        return true;
    }

    if (value == "0" || value == "false" || value == "FALSE" || value == "False") {
        return false;
    }

    throw std::runtime_error{
            std::format("Invalid bool value '{}' at line {}, column {}", value, line_number, column_number)};
}

[[nodiscard]] inline int parse_fixed_int_part(const std::string_view value,
                                              const std::size_t offset,
                                              const std::size_t length,
                                              const std::size_t line_number,
                                              const std::size_t column_number,
                                              const std::string_view part_name) {
    auto result = int{};

    const auto *begin = value.data() + offset;
    const auto *end = begin + length;

    const auto [ptr, error] = std::from_chars(begin, end, result);

    if (error != std::errc{} || ptr != end) {
        throw std::runtime_error{std::format("Invalid date {} in value '{}' at line {}, column {}",
                                             part_name,
                                             value,
                                             line_number,
                                             column_number)};
    }

    return result;
}

[[nodiscard]] inline std::chrono::year_month_day parse_year_month_day_cell(const std::string &value,
                                                                           const std::size_t line_number,
                                                                           const std::size_t column_number) {
    if (value.size() != 10 || value[4] != '-' || value[7] != '-') {
        throw std::runtime_error{std::format("Invalid date value '{}'. Expected YYYY-MM-DD at line {}, column {}",
                                             value,
                                             line_number,
                                             column_number)};
    }

    const auto year_value = parse_fixed_int_part(value, 0, 4, line_number, column_number, "year");
    const auto month_value = parse_fixed_int_part(value, 5, 2, line_number, column_number, "month");
    const auto day_value = parse_fixed_int_part(value, 8, 2, line_number, column_number, "day");

    const auto date = std::chrono::year{year_value} / std::chrono::month{static_cast<unsigned>(month_value)} /
                      std::chrono::day{static_cast<unsigned>(day_value)};

    if (!date.ok()) {
        throw std::runtime_error{
                std::format("Invalid calendar date '{}' at line {}, column {}", value, line_number, column_number)};
    }

    return date;
}

template<typename T>
[[nodiscard]] T parse_required_cell(const raw_csv_cell_s &cell,
                                    const std::size_t line_number,
                                    const std::size_t column_number) {
    if (cell.value.empty()) {
        if constexpr (std::same_as<T, std::string>) {
            if (cell.quoted) {
                return std::string{};
            }
        }

        throw std::runtime_error{std::format("Empty required value at line {}, column {}", line_number, column_number)};
    }

    if constexpr (std::same_as<T, std::string>) {
        return cell.value;
    } else if constexpr (std::same_as<T, bool>) {
        return parse_bool_cell(cell.value, line_number, column_number);
    } else if constexpr (std::integral<T>) {
        return parse_integer_cell<T>(cell.value, line_number, column_number);
    } else if constexpr (std::floating_point<T>) {
        return parse_floating_point_cell<T>(cell.value, line_number, column_number);
    } else if constexpr (is_year_month_day_v<T>) {
        return parse_year_month_day_cell(cell.value, line_number, column_number);
    } else if constexpr (is_sys_days_v<T>) {
        return std::chrono::sys_days{parse_year_month_day_cell(cell.value, line_number, column_number)};
    } else {
        static_assert(always_false_v<T>, "Unsupported CSV column type");
    }

    std::unreachable();
}

template<typename T>
[[nodiscard]] std::optional<T> parse_nullable_cell(const raw_csv_cell_s &cell,
                                                   const std::size_t line_number,
                                                   const std::size_t column_number) {
    if (cell.value.empty() && !cell.quoted) {
        return std::nullopt;
    }

    return parse_required_cell<T>(cell, line_number, column_number);
}

template<typename... ColumnTypes, std::size_t... Indexes>
[[nodiscard]] csv_row_t<ColumnTypes...> parse_required_row_impl(const raw_csv_row_s &raw_row,
                                                                std::index_sequence<Indexes...>) {
    return csv_row_t<ColumnTypes...>{
            parse_required_cell<ColumnTypes>(raw_row.cells[Indexes], raw_row.line_number, Indexes + 1)...};
}

template<typename... ColumnTypes>
[[nodiscard]] csv_row_t<ColumnTypes...> parse_required_row(const raw_csv_row_s &raw_row) {
    constexpr auto expected_column_count = sizeof...(ColumnTypes);

    if (raw_row.cells.size() != expected_column_count) {
        throw std::runtime_error{std::format("Invalid column count at line {}. Expected {}, got {}",
                                             raw_row.line_number,
                                             expected_column_count,
                                             raw_row.cells.size())};
    }

    return parse_required_row_impl<ColumnTypes...>(raw_row, std::index_sequence_for<ColumnTypes...>{});
}

template<typename... ColumnTypes, std::size_t... Indexes>
[[nodiscard]] nullable_csv_row_t<ColumnTypes...> parse_nullable_row_impl(const raw_csv_row_s &raw_row,
                                                                         std::index_sequence<Indexes...>) {
    return nullable_csv_row_t<ColumnTypes...>{
            parse_nullable_cell<ColumnTypes>(raw_row.cells[Indexes], raw_row.line_number, Indexes + 1)...};
}

template<typename... ColumnTypes>
[[nodiscard]] nullable_csv_row_t<ColumnTypes...> parse_nullable_row(const raw_csv_row_s &raw_row) {
    constexpr auto expected_column_count = sizeof...(ColumnTypes);

    if (raw_row.cells.size() != expected_column_count) {
        throw std::runtime_error{std::format("Invalid column count at line {}. Expected {}, got {}",
                                             raw_row.line_number,
                                             expected_column_count,
                                             raw_row.cells.size())};
    }

    return parse_nullable_row_impl<ColumnTypes...>(raw_row, std::index_sequence_for<ColumnTypes...>{});
}

} // namespace ml::util::detail