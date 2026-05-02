#include "csv_load.hpp"

#include <cstdint>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <filesystem>
#include <vector>

#include <algorithm>
#include <ranges>
#include <system_error>
#include <limits>

namespace ml::util::detail {

char resolve_delimiter(const std::optional<char> delimiter) {
    const auto actual_delimiter = delimiter.value_or(',');

    if (actual_delimiter == '\0' || actual_delimiter == '\n' || actual_delimiter == '\r' || actual_delimiter == '"') {
        throw std::runtime_error{"Invalid CSV delimiter"};
    }

    return actual_delimiter;
}

bool is_space(const char c) noexcept { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

bool is_empty_or_spaces(const std::string_view line) noexcept {
    return std::ranges::all_of(line, [](const char ch) noexcept { return is_space(ch); });
}

std::string trim(const std::string_view value) {
    auto begin = std::size_t{0};

    while (begin < value.size() && is_space(value[begin])) {
        ++begin;
    }

    auto end = value.size();

    while (end > begin && is_space(value[end - 1])) {
        --end;
    }

    return std::string{value.substr(begin, end - begin)};
}

std::uintmax_t get_file_size(const std::filesystem::path &path) {
    const auto path_string = path.string();

    auto error = std::error_code{};
    const auto file_size = std::filesystem::file_size(path, error);

    if (error) {
        throw std::runtime_error{std::format("Cannot get file size: '{}'. Reason: {}", path_string, error.message())};
    }

    return file_size;
}

unique_file_ptr_t open_file_for_reading(const std::filesystem::path &path) {
    const auto path_string = path.string();

    auto *file = std::fopen(path_string.c_str(), "rb");

    if (file == nullptr) {
        throw std::runtime_error{std::format("Cannot open file: '{}'. Reason: {}", path_string, std::strerror(errno))};
    }

    return unique_file_ptr_t{file};
}

std::string read_file_content(const std::filesystem::path &path) {
    const auto path_string = path.string();

    auto file = open_file_for_reading(path);
    const auto file_size = get_file_size(path);

    if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error{std::format("File is too large: '{}'", path_string)};
    }

    auto content = std::string{};
    content.resize(static_cast<std::size_t>(file_size));

    if (content.empty()) {
        return content;
    }

    const auto read_bytes = std::fread(content.data(), sizeof(char), content.size(), file.get());

    if (read_bytes != content.size()) {
        if (std::ferror(file.get()) != 0) {
            throw std::runtime_error{std::format("Error while reading file: '{}'", path_string)};
        }

        content.resize(read_bytes);
    }

    return content;
}

std::vector<raw_csv_cell_s> split_csv_record(const std::string_view record,
                                             const char delimiter,
                                             const std::size_t line_number) {
    auto cells = std::vector<raw_csv_cell_s>{};

    auto cell = std::string{};
    auto in_quotes = false;
    auto cell_was_quoted = false;
    auto quote_was_closed = false;

    const auto push_cell = [&cells, &cell, &cell_was_quoted, &quote_was_closed] {
        if (cell_was_quoted) {
            cells.push_back(raw_csv_cell_s{.value = cell, .quoted = true});
        } else {
            cells.push_back(raw_csv_cell_s{.value = trim(cell), .quoted = false});
        }

        cell.clear();
        cell_was_quoted = false;
        quote_was_closed = false;
    };

    for (auto i = std::size_t{0}; i < record.size(); ++i) {
        const auto ch = record[i];

        if (in_quotes) {
            if (ch == '"') {
                in_quotes = false;
                quote_was_closed = true;
            } else {
                cell.push_back(ch);
            }

            continue;
        }

        if (quote_was_closed) {
            if (ch == delimiter) {
                push_cell();
                continue;
            }

            if (is_space(ch)) {
                continue;
            }

            throw std::runtime_error{
                    std::format("Unexpected character '{}' after closing quote at line {}", ch, line_number)};
        }

        if (ch == delimiter) {
            push_cell();
            continue;
        }

        if (ch == '"') {
            if (!is_empty_or_spaces(cell)) {
                throw std::runtime_error{std::format("Unexpected quote inside unquoted field at line {}", line_number)};
            }

            cell.clear();
            in_quotes = true;
            cell_was_quoted = true;
            continue;
        }

        cell.push_back(ch);
    }

    if (in_quotes) {
        throw std::runtime_error{std::format("Unclosed quote in CSV record at line {}", line_number)};
    }

    push_cell();

    return cells;
}

csv_headers_t make_headers(const std::vector<raw_csv_cell_s> &cells, const std::size_t line_number) {
    auto headers = csv_headers_t{};
    headers.reserve(cells.size());

    for (const auto &[i, cell] : std::views::enumerate(cells)) {
        if (cell.value.empty()) {
            throw std::runtime_error{std::format("Empty CSV header at line {}, column {}", line_number, i + 1)};
        }

        headers.push_back(cell.value);
    }

    return headers;
}

raw_csv_content_s parse_raw_csv(const std::filesystem::path &path, const char delimiter, const bool has_headers) {
    const auto path_string = path.string();
    const auto content = read_file_content(path);

    if (content.empty()) {
        throw std::runtime_error{std::format("CSV file is empty: '{}'", path_string)};
    }

    auto result = raw_csv_content_s{};

    auto line_start = std::size_t{0};
    auto line_number = std::size_t{0};
    auto first_content_line_was_read = false;

    while (line_start <= content.size()) {
        auto line_end = content.find('\n', line_start);

        if (line_end == std::string::npos) {
            line_end = content.size();
        }

        auto line = std::string_view{content.data() + line_start, line_end - line_start};

        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        ++line_number;

        if (!is_empty_or_spaces(line)) {
            const auto cells = split_csv_record(line, delimiter, line_number);

            if (!first_content_line_was_read && has_headers) {
                result.headers = make_headers(cells, line_number);
            } else {
                result.rows.push_back(raw_csv_row_s{.line_number = line_number, .cells = cells});
            }

            first_content_line_was_read = true;
        }

        if (line_end == content.size()) {
            break;
        }

        line_start = line_end + 1;
    }

    if (has_headers && !result.headers.has_value()) {
        throw std::runtime_error{std::format("CSV header was not found in file '{}'", path_string)};
    }

    if (result.rows.empty()) {
        throw std::runtime_error{std::format("CSV contains no data rows: '{}'", path_string)};
    }

    return result;
}

} // namespace ml::util::detail