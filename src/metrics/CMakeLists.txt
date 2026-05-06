target_sources(
        ml PRIVATE
        regression/mean_absolute_error.hpp
        regression/mean_squared_error.hpp
        regression/metrics.hpp
        regression/r2_score.hpp
        regression/root_mean_squared_error.hpp)

target_sources(
        ml PRIVATE
        generalization.cpp
        generalization.hpp)