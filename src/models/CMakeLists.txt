target_sources(
        ml PRIVATE
        detail/knn_base.cpp
        detail/knn_base.hpp
        detail/linear_scoring_model_base.cpp
        detail/linear_scoring_model_base.hpp)

target_sources(
        ml PRIVATE
        knn_classifier.cpp
        knn_classifier.hpp
        knn_regressor.cpp
        knn_regressor.hpp
        linear_regression.cpp
        linear_regression.hpp
        logistic_regression.cpp
        logistic_regression.hpp)