cmake_minimum_required(VERSION 3.30)

project(
    ml
    VERSION 1.0.1
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)

add_executable(ml src/main.cpp)

target_include_directories(ml PUBLIC src)

target_compile_definitions(ml PRIVATE ML_APPLICATION_DIR_PATH="${PROJECT_SOURCE_DIR}")

include(FetchContent)

FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Eigen3)

target_link_libraries(ml
        PRIVATE
        Eigen3::Eigen
)

FetchContent_Declare(
        csv
        GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
        GIT_TAG        4.0.0
        GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(csv)

target_link_libraries(ml
        PRIVATE
        Eigen3::Eigen
        csv
)

add_subdirectory(src)