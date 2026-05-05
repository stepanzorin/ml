#include <csv.hpp>

#include "models/linear_regression.hpp"

int main() {
    auto samples = std::vector<ml::common::sample_s<double>>{{{15.0, 0.0, 1.0}, 28.0},
                                                             {{16.0, 0.0, 1.0}, 30.0},
                                                             {{17.0, 0.0, 2.0}, 34.0},
                                                             {{18.0, 0.0, 2.0}, 37.0},
                                                             {{19.0, 0.0, 2.0}, 40.0},

                                                             {{20.0, 1.0, 2.0}, 50.0},
                                                             {{21.0, 1.0, 3.0}, 56.0},
                                                             {{22.0, 1.0, 3.0}, 60.0},
                                                             {{23.0, 1.0, 3.0}, 64.0},
                                                             {{24.0, 1.0, 4.0}, 70.0}};

    auto model = ml::models::LinearRegression(3);

    model.train(samples, 20'000, 0.0001);

    model.print_parameters();

    const auto new_day = std::vector{27.0, 1.0, 5.0};

    const auto prediction = model.predict(new_day);

    std::cout << "\nPrediction: " << prediction << '\n';

    return 0;
}