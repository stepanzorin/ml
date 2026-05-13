target_sources(
        ml PRIVATE
        binary_classification/confusion_matrix.cpp
        binary_classification/confusion_matrix.hpp
        binary_classification/log_loss.cpp
        binary_classification/log_loss.hpp)

target_sources(
        ml PRIVATE
        regression/mean_absolute_error.hpp
        regression/mean_squared_error.hpp
        regression/r2_score.hpp
        regression/root_mean_squared_error.hpp)

target_sources(
        ml PRIVATE
        generalization.cpp
        generalization.hpp
        metrics.hpp)