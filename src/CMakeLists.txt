add_subdirectory(algorithms)
add_subdirectory(common)
add_subdirectory(metrics)
add_subdirectory(models)
add_subdirectory(regularization)
add_subdirectory(util)

target_sources(
        ml PRIVATE
        dataset_loader.cpp
        dataset_loader.hpp
        types.hpp)