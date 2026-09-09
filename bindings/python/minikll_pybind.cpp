#include <minikll/minikll.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using minikll::CompactMode;
using minikll::OddPolicy;
using minikll::SketchArrays;
using minikll::WeightedSample;
using minikll::WeightedSketch1D;
using minikll::weight_type;

std::vector<WeightedSample> samples_from_pairs(
    const std::vector<std::pair<double, weight_type>>& samples) {
  std::vector<WeightedSample> out;
  out.reserve(samples.size());
  for (const auto& sample : samples) {
    out.push_back(WeightedSample{sample.first, sample.second});
  }
  return out;
}

py::tuple arrays_to_tuple(const SketchArrays& arrays) {
  return py::make_tuple(arrays.values, arrays.weights);
}

std::vector<std::pair<double, weight_type>> items_to_pairs(
    const WeightedSketch1D& sketch) {
  std::vector<std::pair<double, weight_type>> out;
  out.reserve(sketch.items().size());
  for (const auto& item : sketch.items()) {
    out.emplace_back(item.value, item.weight);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(minikll, m) {
  m.doc() = "Minimal C++ KLL-like quantile sketches";

  py::enum_<CompactMode>(m, "CompactMode")
      .value("Deterministic", CompactMode::Deterministic)
      .value("Random", CompactMode::Random)
      .export_values();

  py::enum_<OddPolicy>(m, "OddPolicy")
      .value("Carry", OddPolicy::Carry)
      .value("DropLargest", OddPolicy::DropLargest)
      .export_values();

  py::class_<WeightedSample>(m, "WeightedSample")
      .def(py::init<double, weight_type>(), py::arg("value"), py::arg("weight"))
      .def_readwrite("value", &WeightedSample::value)
      .def_readwrite("weight", &WeightedSample::weight)
      .def("__repr__", [](const WeightedSample& sample) {
        return "WeightedSample(value=" + std::to_string(sample.value) +
            ", weight=" + std::to_string(sample.weight) + ")";
      });

  py::class_<WeightedSketch1D>(m, "WeightedSketch1D")
      .def(py::init<std::size_t>(), py::arg("s"))
      .def_static(
          "from_samples",
          [](const std::vector<std::pair<double, weight_type>>& samples,
             std::size_t s,
             CompactMode mode,
             std::uint64_t seed,
             OddPolicy odd_policy) {
            return WeightedSketch1D::from_samples(
                samples_from_pairs(samples), s, mode, seed, odd_policy);
          },
          py::arg("samples"),
          py::arg("s"),
          py::arg("mode") = CompactMode::Random,
          py::arg("seed") = 0,
          py::arg("odd_policy") = OddPolicy::Carry,
          "Build from ``[(value, weight), ...]`` samples.")
      .def_static(
          "from_sorted_values",
          &WeightedSketch1D::from_sorted_values,
          py::arg("values_sorted"),
          py::arg("s"),
          py::arg("mode") = CompactMode::Random,
          py::arg("seed") = 0,
          py::arg("odd_policy") = OddPolicy::Carry,
          "Build from unit-weight values. Input is sorted if needed.")
      .def_static(
          "from_arrays",
          &WeightedSketch1D::from_arrays,
          py::arg("values"),
          py::arg("weights"),
          py::arg("s") = 0,
          "Import value and weight arrays. Zero weights are ignored.")
      .def_property_readonly("size_limit", &WeightedSketch1D::size_limit)
      .def_property_readonly("retained", &WeightedSketch1D::retained)
      .def_property_readonly("empty", &WeightedSketch1D::empty)
      .def_property_readonly("total_weight", &WeightedSketch1D::total_weight)
      .def(
          "items",
          &items_to_pairs,
          "Return active retained items as ``[(value, weight), ...]``.")
      .def(
          "to_arrays",
          [](const WeightedSketch1D& sketch, bool fixed_size) {
            return arrays_to_tuple(sketch.to_arrays(fixed_size));
          },
          py::arg("fixed_size") = false,
          "Return ``(values, weights)``. Fixed-size padding is opt-in.")
      .def(
          "to_padded_arrays",
          [](const WeightedSketch1D& sketch) {
            return arrays_to_tuple(sketch.to_padded_arrays());
          },
          "Return fixed-size ``(values, weights)`` padded with ``(+inf, 0)``.")
      .def("quantile", &WeightedSketch1D::quantile, py::arg("q"))
      .def(
          "quantiles",
          &WeightedSketch1D::quantiles,
          py::arg("qs"),
          "Return quantiles for all ranks in ``qs`` with one retained-item scan.")
      .def("median", &WeightedSketch1D::median)
      .def(
          "merge",
          [](const WeightedSketch1D& sketch,
             const WeightedSketch1D& other,
             CompactMode mode,
             std::uint64_t seed,
             bool fixed_size,
             OddPolicy odd_policy) {
            return sketch.merge(other, mode, seed, fixed_size, odd_policy);
          },
          py::arg("other"),
          py::arg("mode") = CompactMode::Random,
          py::arg("seed") = 0,
          py::arg("fixed_size") = false,
          py::arg("odd_policy") = OddPolicy::Carry);

  py::class_<minikll::KLLSketch1D>(m, "KLLSketch1D")
      .def(
          py::init<
              std::size_t,
              std::size_t,
              CompactMode,
              std::uint64_t,
              bool,
              OddPolicy>(),
          py::arg("k"),
          py::arg("s"),
          py::arg("mode") = CompactMode::Random,
          py::arg("seed") = 0,
          py::arg("ignore_nan") = true,
          py::arg("odd_policy") = OddPolicy::Carry)
      .def("update", &minikll::KLLSketch1D::update, py::arg("x"))
      .def("update_many", &minikll::KLLSketch1D::update_many, py::arg("xs"))
      .def(
          "update_batch",
          &minikll::KLLSketch1D::update_batch,
          py::arg("xs"),
          py::arg("pre_sorted") = false,
          "Insert a materialized batch with one append/compaction pass.")
      .def(
          "update_sorted_batch",
          &minikll::KLLSketch1D::update_sorted_batch,
          py::arg("xs"),
          "Convenience wrapper for update_batch(xs, pre_sorted=True).")
      .def(
          "finalize",
          &minikll::KLLSketch1D::finalize,
          py::arg("fixed_size") = false)
      .def_property_readonly("count", &minikll::KLLSketch1D::count)
      .def_property_readonly("k", &minikll::KLLSketch1D::k)
      .def_property_readonly("size_limit", &minikll::KLLSketch1D::size_limit)
      .def(
          "normalized_rank_error",
          &minikll::KLLSketch1D::normalized_rank_error,
          py::arg("pmf") = false)
      .def_property_readonly("levels", &minikll::KLLSketch1D::levels);

  py::class_<minikll::KLLSketchND>(m, "KLLSketchND")
      .def(
          py::init<
              std::size_t,
              std::size_t,
              std::size_t,
              CompactMode,
              std::uint64_t,
              bool,
              OddPolicy>(),
          py::arg("k"),
          py::arg("s"),
          py::arg("d"),
          py::arg("mode") = CompactMode::Random,
          py::arg("seed") = 0,
          py::arg("ignore_nan") = true,
          py::arg("odd_policy") = OddPolicy::Carry)
      .def("update_row", &minikll::KLLSketchND::update_row, py::arg("row"))
      .def(
          "update_matrix",
          &minikll::KLLSketchND::update_matrix,
          py::arg("matrix"))
      .def(
          "finalize",
          &minikll::KLLSketchND::finalize,
          py::arg("fixed_size") = false)
      .def_property_readonly("dimension", &minikll::KLLSketchND::dimension);

  m.def(
      "compact_mode_from_string",
      &minikll::compact_mode_from_string,
      py::arg("mode"));
  m.def(
      "compact_mode_to_string",
      &minikll::compact_mode_to_string,
      py::arg("mode"));
  m.def(
      "odd_policy_from_string",
      &minikll::odd_policy_from_string,
      py::arg("policy"));
  m.def(
      "odd_policy_to_string",
      &minikll::odd_policy_to_string,
      py::arg("policy"));
  m.def(
      "build_sketch_from_sorted_values",
      [](const std::vector<double>& values_sorted,
         std::size_t s,
         CompactMode mode,
         std::uint64_t seed,
         bool fixed_size,
         OddPolicy odd_policy) {
        return arrays_to_tuple(minikll::build_sketch_from_sorted_values(
            values_sorted, s, mode, seed, fixed_size, odd_policy));
      },
      py::arg("values_sorted"),
      py::arg("s"),
      py::arg("mode") = CompactMode::Random,
      py::arg("seed") = 0,
      py::arg("fixed_size") = false,
      py::arg("odd_policy") = OddPolicy::Carry);
  m.def(
      "build_sketch_from_column",
      [](std::vector<double> column,
         std::size_t s,
         bool pre_sorted,
         CompactMode mode,
         std::uint64_t seed,
         bool fixed_size,
         OddPolicy odd_policy) {
        return arrays_to_tuple(minikll::build_sketch_from_column(
            std::move(column),
            s,
            pre_sorted,
            mode,
            seed,
            fixed_size,
            odd_policy));
      },
      py::arg("column"),
      py::arg("s"),
      py::arg("pre_sorted") = false,
      py::arg("mode") = CompactMode::Random,
      py::arg("seed") = 0,
      py::arg("fixed_size") = false,
      py::arg("odd_policy") = OddPolicy::Carry);
  m.def(
      "merge_two_sorted_weighted",
      [](const std::vector<double>& a_values,
         const std::vector<weight_type>& a_weights,
         const std::vector<double>& b_values,
         const std::vector<weight_type>& b_weights,
         bool drop_zero,
         bool combine_equal) {
        return arrays_to_tuple(minikll::merge_two_sorted_weighted(
            a_values, a_weights, b_values, b_weights, drop_zero, combine_equal));
      },
      py::arg("a_values"),
      py::arg("a_weights"),
      py::arg("b_values"),
      py::arg("b_weights"),
      py::arg("drop_zero") = true,
      py::arg("combine_equal") = false);
  m.def(
      "merge_k_sorted_lists",
      [](const std::vector<std::vector<double>>& values_list,
         const std::vector<std::vector<weight_type>>& weights_list,
         bool drop_zero,
         bool combine_equal) {
        return arrays_to_tuple(minikll::merge_k_sorted_lists(
            values_list, weights_list, drop_zero, combine_equal));
      },
      py::arg("values_list"),
      py::arg("weights_list"),
      py::arg("drop_zero") = true,
      py::arg("combine_equal") = false);
  m.def(
      "weighted_median_from_sorted_weighted",
      &minikll::weighted_median_from_sorted_weighted,
      py::arg("values"),
      py::arg("weights"));
  m.def(
      "weighted_median_from_k_sorted_sketches",
      &minikll::weighted_median_from_k_sorted_sketches,
      py::arg("values_list"),
      py::arg("weights_list"),
      py::arg("drop_zero") = true,
      py::arg("combine_equal") = false);
  m.def(
      "weighted_median_from_arrays",
      &minikll::weighted_median_from_arrays,
      py::arg("values"),
      py::arg("weights"));
  m.def(
      "merge_weighted_sketches",
      &minikll::merge_weighted_sketches,
      py::arg("sketches"),
      py::arg("mode") = CompactMode::Random,
      py::arg("seed") = 0,
      py::arg("fixed_size") = false,
      py::arg("odd_policy") = OddPolicy::Carry);
  m.def(
      "normalized_rank_error",
      &minikll::normalized_rank_error,
      py::arg("k"),
      py::arg("pmf") = false);
  m.def("suggest_k", &minikll::suggest_k, py::arg("epsilon"), py::arg("pmf") = false);
}
