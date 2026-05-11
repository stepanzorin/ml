cmake_minimum_required(VERSION 3.30)

project(
        ml
        VERSION 1.0.1
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ml src/main.cpp)

target_include_directories(ml PUBLIC src)

target_compile_definitions(ml PRIVATE ML_APPLICATION_DIR_PATH="${PROJECT_SOURCE_DIR}")

include(FetchContent)

FetchContent_Declare(
        rapidjson
        GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
        GIT_TAG        v1.1.0
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR include
)

FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(rapidjson)
FetchContent_MakeAvailable(Eigen3)
FetchContent_MakeAvailable(spdlog)

target_include_directories(ml
        SYSTEM PRIVATE
        ${rapidjson_SOURCE_DIR}/include
)

target_compile_definitions(ml
        PRIVATE
        RAPIDJSON_HAS_STDSTRING=1
)

target_link_libraries(ml
        PRIVATE
        Eigen3::Eigen
        spdlog::spdlog
)

add_subdirectory(src)