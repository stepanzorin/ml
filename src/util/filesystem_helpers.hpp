#pragma once

#include <filesystem>

namespace ml::util {

[[nodiscard]] inline std::filesystem::path application_directory_path() noexcept { return ML_APPLICATION_DIR_PATH; }

[[nodiscard]] inline std::filesystem::path datasets_directory_path() noexcept {
    return application_directory_path() / "datasets";
}

} // namespace ml::util