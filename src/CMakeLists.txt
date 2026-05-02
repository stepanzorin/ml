target_sources(
        ml PRIVATE
        csv.cpp
        csv.hpp)

add_subdirectory(common)
add_subdirectory(models)
add_subdirectory(util)