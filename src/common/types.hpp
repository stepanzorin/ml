// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>

#pragma once

// TODO: Current types wrote for work with CVS datasets. (may be will support other formats. may be not)
//       Current types wrote only for the `Linear Regression` method.
//       It's all will be extended for another methods in the future.

#include <string>
#include <vector>

#include "common/concepts.hpp"

namespace ml::common {

template<std::default_initializable TargetType>
struct sample_s {
    std::vector<double> features;
    TargetType target = {};
};

template<dataset_sample SampleType, bool Normalized = false>
struct tabular_dataset_s {
    std::vector<std::string> feature_names;
    std::string target_name;
    std::vector<SampleType> samples;
};

} // namespace ml::common