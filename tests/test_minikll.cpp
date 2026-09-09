/**
 * @file test_minikll.cpp
 *
 * Lightweight executable tests for the public minikll API. The tests use plain
 * `assert` so the library can be verified without bringing in a test framework.
 */

#include <minikll/minikll.hpp>

// The tests express their checks with assert(), so assertions must stay live
// even when the project is built with -DNDEBUG (e.g. CMAKE_BUILD_TYPE=Release).
// Undefine NDEBUG before <cassert> so the suite always actually verifies results.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using minikll::CompactMode;
using minikll::WeightedSample;
using minikll::WeightedSketch1D;
using minikll::weight_type;

/** Compare doubles exactly, with explicit support for +inf padding sentinels. */
void assert_double_equal(double actual, double expected) {
  if (std::isinf(expected)) {
    assert(std::isinf(actual) && actual > 0);
  } else {
    assert(actual == expected);
  }
}

/** Assert that two value arrays match, including any +inf padding entries. */
void assert_values_equal(
    const std::vector<double>& actual,
    const std::vector<double>& expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    assert_double_equal(actual[i], expected[i]);
  }
}

/** Assert exact equality for integer weight arrays. */
void assert_weights_equal(
    const std::vector<weight_type>& actual,
    const std::vector<weight_type>& expected) {
  assert(actual == expected);
}

/** Assert approximate scalar equality for formulas involving floating point. */
void assert_close(double actual, double expected, double tol = 1e-9) {
  assert(std::fabs(actual - expected) <= tol);
}

/** Assert that a callable throws some standard exception. */
template <typename Fn>
void assert_throws(Fn&& fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception&) {
    threw = true;
  }
  assert(threw);
}

/** Append `count` +inf values to make expected padded arrays concise. */
std::vector<double> with_inf(std::vector<double> values, std::size_t count) {
  values.insert(
      values.end(), count, std::numeric_limits<double>::infinity());
  return values;
}

/** Construction should sort, combine duplicate values, and pad only on request. */
void test_weighted_from_samples_exact_sort_combine_pad() {
  const std::vector<WeightedSample> samples = {
      {5.0, 1}, {1.0, 2}, {3.0, 1}, {1.0, 3}};
  const auto sketch =
      WeightedSketch1D::from_samples(samples, 6, CompactMode::Deterministic);
  const auto arrays = sketch.to_arrays();

  assert_values_equal(arrays.values, {1.0, 3.0, 5.0});
  assert_weights_equal(arrays.weights, {5, 1, 1});

  const auto padded = sketch.to_arrays(true);
  assert_values_equal(padded.values, with_inf({1.0, 3.0, 5.0}, 3));
  assert_weights_equal(padded.weights, {5, 1, 1, 0, 0, 0});
}

/** Weighted quantiles should follow the inclusive cumulative-weight convention. */
void test_weighted_quantile_expected_values() {
  const auto sketch = WeightedSketch1D::from_samples(
      {{1.0, 2}, {2.0, 3}, {10.0, 1}}, 8);

  assert_close(sketch.quantile(0.0), 1.0);
  assert_close(sketch.quantile(0.25), 1.0);
  assert_close(sketch.quantile(0.5), 2.0);
  assert_close(sketch.quantile(0.75), 2.0);
  assert_close(sketch.quantile(1.0), 10.0);
}

/** Batch quantiles should scan retained items once and preserve query order. */
void test_weighted_quantiles_batch_query() {
  const auto sketch = WeightedSketch1D::from_samples(
      {{1.0, 2}, {2.0, 3}, {10.0, 1}}, 8);

  const std::vector<double> qs = {0.75, 0.0, 1.0, 0.5};
  const auto got = sketch.quantiles(qs);
  assert_values_equal(got, {2.0, 1.0, 10.0, 2.0});
  for (std::size_t i = 0; i < qs.size(); ++i) {
    assert_close(got[i], sketch.quantile(qs[i]));
  }

  assert(sketch.quantiles({}).empty());
  assert_throws([&]() { (void)sketch.quantiles({0.5, 1.1}); });

  const WeightedSketch1D empty(4);
  assert_throws([&]() { (void)empty.quantiles({0.5}); });
}

/** Even-size medians use the lower/inclusive median convention. */
void test_even_quantile_uses_inclusive_lower_median() {
  const auto sketch =
      WeightedSketch1D::from_samples({{1.0, 1}, {2.0, 1}}, 4);

  assert_close(sketch.quantile(0.0), 1.0);
  assert_close(sketch.quantile(0.5), 1.0);
  assert_close(sketch.median(), 1.0);
  assert_close(sketch.quantile(0.500001), 2.0);
  assert_close(sketch.quantile(1.0), 2.0);
}

/** Array import should ignore zero-weight padding, sort, and combine duplicates. */
void test_from_arrays_sorts_combines_and_ignores_zero_padding() {
  const std::vector<double> values = {
      std::numeric_limits<double>::infinity(), 3.0, 1.0, 3.0, 2.0};
  const std::vector<weight_type> weights = {0, 2, 4, 5, 0};

  const auto sketch = WeightedSketch1D::from_arrays(values, weights, 4);
  const auto arrays = sketch.to_arrays();
  assert_values_equal(arrays.values, {1.0, 3.0});
  assert_weights_equal(arrays.weights, {4, 7});

  const auto padded = sketch.to_padded_arrays();
  assert_values_equal(
      padded.values,
      {1.0, 3.0, std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()});
  assert_weights_equal(padded.weights, {4, 7, 0, 0});
  assert_close(sketch.median(), 3.0);
}

/** Deterministic compaction keeps the right item from each adjacent pair. */
void test_weighted_compaction_deterministic_expected() {
  std::vector<WeightedSample> samples;
  for (int i = 1; i <= 8; ++i) {
    samples.push_back({static_cast<double>(i), 1});
  }

  const auto sketch =
      WeightedSketch1D::from_samples(samples, 4, CompactMode::Deterministic);
  auto arrays = sketch.to_arrays();
  assert_values_equal(arrays.values, {2.0, 4.0, 6.0, 8.0});
  assert_weights_equal(arrays.weights, {2, 2, 2, 2});

  const auto sketch2 =
      WeightedSketch1D::from_samples(samples, 2, CompactMode::Deterministic);
  arrays = sketch2.to_arrays();
  assert_values_equal(arrays.values, {4.0, 8.0});
  assert_weights_equal(arrays.weights, {4, 4});
}

/** The default odd-item policy carries one unpaired item to preserve weight. */
void test_odd_compaction_carries_unpaired_item() {
  std::vector<WeightedSample> samples;
  for (int i = 1; i <= 5; ++i) {
    samples.push_back({static_cast<double>(i), 1});
  }

  const auto sketch =
      WeightedSketch1D::from_samples(samples, 3, CompactMode::Deterministic);
  const auto arrays = sketch.to_arrays();
  assert_values_equal(arrays.values, {2.0, 4.0, 5.0});
  assert_weights_equal(arrays.weights, {2, 2, 1});
  assert(sketch.total_weight() == 5);

  const auto padded = sketch.to_padded_arrays();
  assert_values_equal(padded.values, {2.0, 4.0, 5.0});
  assert_weights_equal(padded.weights, {2, 2, 1});
}

/** Legacy odd-item handling drops the largest unpaired item explicitly. */
void test_odd_compaction_drop_largest_legacy_policy() {
  std::vector<WeightedSample> samples;
  for (int i = 1; i <= 5; ++i) {
    samples.push_back({static_cast<double>(i), 1});
  }

  const auto sketch = WeightedSketch1D::from_samples(
      samples,
      3,
      CompactMode::Deterministic,
      0,
      minikll::OddPolicy::DropLargest);
  const auto arrays = sketch.to_arrays();
  assert_values_equal(arrays.values, {2.0, 4.0});
  assert_weights_equal(arrays.weights, {2, 2});
  assert(sketch.total_weight() == 4);
}

/** Equal retained values can be combined because only their total weight matters. */
void test_sorted_value_duplicate_compaction_combines_retained_values() {
  const auto arrays =
      minikll::build_sketch_from_sorted_values({1.0, 1.0, 1.0, 1.0}, 2);

  assert_values_equal(arrays.values, {1.0});
  assert_weights_equal(arrays.weights, {4});

  const auto padded =
      minikll::build_sketch_from_sorted_values(
          {1.0, 1.0, 1.0, 1.0}, 2, CompactMode::Deterministic, 0, true);
  assert_values_equal(
      padded.values, {1.0, std::numeric_limits<double>::infinity()});
  assert_weights_equal(padded.weights, {4, 0});
}

/** Seeded random compaction should be reproducible for a fixed seed. */
void test_weighted_compaction_random_seed_reproducible() {
  std::vector<WeightedSample> samples;
  for (int i = 0; i < 100; ++i) {
    samples.push_back({static_cast<double>(i), 1});
  }

  const auto a =
      WeightedSketch1D::from_samples(samples, 16, CompactMode::Random, 123);
  const auto b =
      WeightedSketch1D::from_samples(samples, 16, CompactMode::Random, 123);
  const auto c =
      WeightedSketch1D::from_samples(samples, 16, CompactMode::Random, 456);

  const auto av = a.to_arrays();
  const auto bv = b.to_arrays();
  const auto cv = c.to_arrays();

  assert_values_equal(av.values, bv.values);
  assert_weights_equal(av.weights, bv.weights);
  assert(av.values != cv.values || av.weights != cv.weights);
}

/** Column builders should remain safe even when callers incorrectly mark sorted input. */
void test_build_sketch_from_column_sorts_defensively() {
  const auto arrays =
      minikll::build_sketch_from_column(
          {4.0, 1.0, 3.0, 2.0},
          2,
          false,
          CompactMode::Deterministic);
  assert_values_equal(arrays.values, {2.0, 4.0});
  assert_weights_equal(arrays.weights, {2, 2});

  const auto sorted =
      minikll::build_sketch_from_column({2.0, 1.0}, 2, true);
  assert_values_equal(sorted.values, {1.0, 2.0});
  assert_weights_equal(sorted.weights, {1, 1});
}

/** Sketch merging should produce a new sketch without mutating its inputs. */
void test_merge_expected_and_no_mutation() {
  const auto a = WeightedSketch1D::from_samples(
      {{1.0, 1}, {3.0, 1}, {5.0, 1}}, 8);
  const auto b = WeightedSketch1D::from_samples(
      {{2.0, 1}, {4.0, 1}, {6.0, 1}}, 8);

  const auto a_before = a.to_arrays();
  const auto b_before = b.to_arrays();

  const auto merged = a.merge(b);
  const auto arrays = merged.to_arrays();
  assert_values_equal(
      arrays.values, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  assert_weights_equal(arrays.weights, {1, 1, 1, 1, 1, 1});

  assert_values_equal(a.to_arrays().values, a_before.values);
  assert_weights_equal(a.to_arrays().weights, a_before.weights);
  assert_values_equal(b.to_arrays().values, b_before.values);
  assert_weights_equal(b.to_arrays().weights, b_before.weights);
}

/** The array merge helper should optionally combine adjacent equal values. */
void test_merge_two_combine_equal_flag() {
  const std::vector<double> a_values = {1.0, 3.0};
  const std::vector<weight_type> a_weights = {2, 3};
  const std::vector<double> b_values = {3.0, 4.0};
  const std::vector<weight_type> b_weights = {5, 7};

  const auto separate = minikll::merge_two_sorted_weighted(
      a_values, a_weights, b_values, b_weights, true, false);
  assert_values_equal(separate.values, {1.0, 3.0, 3.0, 4.0});
  assert_weights_equal(separate.weights, {2, 3, 5, 7});

  const auto combined = minikll::merge_two_sorted_weighted(
      a_values, a_weights, b_values, b_weights, true, true);
  assert_values_equal(combined.values, {1.0, 3.0, 4.0});
  assert_weights_equal(combined.weights, {2, 8, 7});
}

/** Zero-weight padding sentinels must not affect weighted medians. */
void test_padding_does_not_affect_median_helper() {
  const std::vector<double> raw_values = {1.0, 2.0, 100.0};
  const std::vector<weight_type> raw_weights = {2, 3, 1};
  const std::vector<double> padded_values = with_inf(raw_values, 2);
  const std::vector<weight_type> padded_weights = {2, 3, 1, 0, 0};

  assert_close(
      minikll::weighted_median_from_arrays(raw_values, raw_weights), 2.0);
  assert_close(
      minikll::weighted_median_from_arrays(raw_values, raw_weights),
      minikll::weighted_median_from_arrays(padded_values, padded_weights));
}

/** Streaming sketches should count non-NaN updates and respect NaN policy. */
void test_kll_1d_basic_and_nan_behavior() {
  minikll::KLLSketch1D sketch(16, 16, CompactMode::Deterministic, 0, true);
  sketch.update_many({3.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0});
  assert(sketch.count() == 3);

  const auto out = sketch.finalize().to_padded_arrays();
  assert_values_equal(out.values, with_inf({1.0, 2.0, 3.0}, 13));
  assert_weights_equal(
      out.weights, {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  minikll::KLLSketch1D error_sketch(
      8, 8, CompactMode::Deterministic, 0, false);
  assert_throws([&]() {
    error_sketch.update(std::numeric_limits<double>::quiet_NaN());
  });
}

/** Batch updates should sort defensively and apply the same NaN policy. */
void test_kll_1d_batch_update_paths() {
  minikll::KLLSketch1D sketch(16, 16, CompactMode::Deterministic, 0, true);
  sketch.update_batch(
      {4.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 3.0, 2.0});
  assert(sketch.count() == 4);

  const auto out = sketch.finalize().to_padded_arrays();
  assert_values_equal(out.values, with_inf({1.0, 2.0, 3.0, 4.0}, 12));
  assert_weights_equal(
      out.weights, {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  minikll::KLLSketch1D sorted(16, 16, CompactMode::Deterministic, 0, true);
  sorted.update_sorted_batch({5.0, 1.0, 4.0, 2.0, 3.0});
  assert(sorted.count() == 5);
  assert_close(sorted.finalize().median(), 3.0);

  minikll::KLLSketch1D error_sketch(
      8, 8, CompactMode::Deterministic, 0, false);
  assert_throws([&]() {
    error_sketch.update_batch({1.0, std::numeric_limits<double>::quiet_NaN()});
  });
}

/** Level-based streaming compaction assigns powers-of-two weights. */
void test_kll_1d_compaction_weight_powers_of_two() {
  minikll::KLLSketch1D sketch(2, 32);
  for (int i = 0; i < 20; ++i) {
    sketch.update(static_cast<double>(i));
  }

  const auto arrays = sketch.finalize().to_arrays();
  bool saw_active = false;
  for (weight_type weight : arrays.weights) {
    if (weight == 0) {
      continue;
    }
    saw_active = true;
    assert((weight & (weight - 1)) == 0);
  }
  assert(saw_active);
}

/** Final compaction is opt-in and should preserve represented weight. */
void test_kll_finalize_fixed_size_is_optional_and_preserves_weight() {
  minikll::KLLSketch1D sketch(4, 3, CompactMode::Deterministic);
  for (int i = 0; i < 25; ++i) {
    sketch.update(static_cast<double>(i));
  }

  const auto active = sketch.finalize(false);
  assert(active.retained() > 3);
  assert(active.size_limit() == 3);
  assert(active.total_weight() == 25);
  assert_throws([&]() { (void)active.to_arrays(true); });

  const auto fixed = sketch.finalize(true);
  assert(fixed.retained() <= 3);
  assert(fixed.size_limit() == 3);
  assert(fixed.total_weight() == 25);
  assert(fixed.to_padded_arrays().values.size() == 3);
}

/** The ND wrapper should sketch columns independently. */
void test_kll_nd_update_and_finalize() {
  const std::vector<std::vector<double>> matrix = {
      {1.0, 10.0, -1.0},
      {2.0, 20.0, -2.0},
      {3.0, 30.0, -3.0},
      {4.0, 40.0, -4.0},
      {5.0, 50.0, -5.0},
  };

  minikll::KLLSketchND nd(16, 16, 3);
  nd.update_matrix(matrix);
  const auto columns = nd.finalize();
  assert(columns.size() == 3);
  assert_close(columns[0].median(), 3.0);
  assert_close(columns[1].median(), 30.0);
  assert_close(columns[2].median(), -3.0);
}

/** Non-fixed merge should collect active retained items and preserve exact mass. */
void test_merge_weighted_sketches_tree_reduction() {
  const std::vector<WeightedSketch1D> sketches = {
      WeightedSketch1D::from_samples({{1.0, 1}, {2.0, 1}}, 16),
      WeightedSketch1D::from_samples({{3.0, 1}, {4.0, 1}}, 16),
      WeightedSketch1D::from_samples({{5.0, 1}, {6.0, 1}}, 16),
      WeightedSketch1D::from_samples({{7.0, 1}, {8.0, 1}}, 16),
  };

  const auto merged = minikll::merge_weighted_sketches(sketches);
  const auto arrays = merged.to_padded_arrays();
  assert_values_equal(
      arrays.values, with_inf({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, 8));
  assert_weights_equal(
      arrays.weights, {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
}

/** Fixed-size merge should compact once after collecting all active items. */
void test_merge_weighted_sketches_compacts_once_when_fixed_size() {
  std::vector<WeightedSketch1D> sketches;
  for (int i = 0; i < 4; ++i) {
    std::vector<WeightedSample> samples;
    for (int j = 0; j < 4; ++j) {
      samples.push_back({static_cast<double>(i * 4 + j), 1});
    }
    sketches.push_back(WeightedSketch1D::from_retained_samples(samples, 4));
  }

  const auto exact = minikll::merge_weighted_sketches(
      sketches, CompactMode::Deterministic, 0, false);
  assert(exact.retained() == 16);
  assert(exact.total_weight() == 16);

  const auto compacted = minikll::merge_weighted_sketches(
      sketches, CompactMode::Deterministic, 0, true);
  assert(compacted.retained() <= 4);
  assert(compacted.size_limit() == 4);
  assert(compacted.total_weight() == 16);
}

/** Fast constructors and weighted-array merge helpers should agree on medians. */
void test_build_sketch_from_sorted_values_and_merge_ops() {
  const auto arrays = minikll::build_sketch_from_sorted_values(
      {1.0, 2.0, 3.0, 4.0}, 2, CompactMode::Deterministic);
  assert_values_equal(arrays.values, {2.0, 4.0});
  assert_weights_equal(arrays.weights, {2, 2});

  const std::vector<std::vector<double>> values = {
      {1.0, 3.0, std::numeric_limits<double>::infinity()},
      {2.0, 4.0, std::numeric_limits<double>::infinity()},
  };
  const std::vector<std::vector<weight_type>> weights = {
      {1, 1, 0},
      {1, 1, 0},
  };
  const auto merged = minikll::merge_k_sorted_lists(values, weights);
  assert_values_equal(merged.values, {1.0, 2.0, 3.0, 4.0});
  assert_weights_equal(merged.weights, {1, 1, 1, 1});
  assert_close(
      minikll::weighted_median_from_k_sorted_sketches(values, weights), 2.0);

  const auto defensively_sorted =
      minikll::build_sketch_from_sorted_values({2.0, 1.0}, 2);
  assert_values_equal(defensively_sorted.values, {1.0, 2.0});
  assert_weights_equal(defensively_sorted.weights, {1, 1});
}

/** Rank-error helpers should expose stable DataSketches-style estimates. */
void test_rank_error_helpers() {
  assert_close(minikll::normalized_rank_error(200, false), 0.0132947574, 1e-9);
  assert_close(minikll::normalized_rank_error(200, true), 0.0165156191, 1e-9);
  assert(minikll::suggest_k(0.0134, false) <= 205);
  assert(minikll::suggest_k(0.0134, false) >= 195);
}

/** Public APIs should reject invalid sizes, empty quantiles, and NaNs. */
void test_error_paths() {
  assert_throws([]() {
    (void)WeightedSketch1D::from_samples({{1.0, 1}}, 0);
  });

  const WeightedSketch1D empty(4);
  assert_throws([&]() { (void)empty.quantile(0.5); });

  const auto a = WeightedSketch1D::from_samples({{1.0, 1}}, 4);
  const auto b = WeightedSketch1D::from_samples({{2.0, 1}}, 8);
  assert_throws([&]() { (void)a.merge(b); });

  assert_throws([]() {
    (void)minikll::merge_weighted_sketches({});
  });

  assert_throws([]() {
    (void)WeightedSketch1D::from_samples(
        {{std::numeric_limits<double>::quiet_NaN(), 1}}, 4);
  });
}

/** KLLSketch1D exposes its export size as size_limit(), matching WeightedSketch1D. */
void test_kll_1d_size_limit_accessor() {
  minikll::KLLSketch1D sketch(16, 64, CompactMode::Deterministic);
  assert(sketch.k() == 16);
  assert(sketch.size_limit() == 64);
  assert(sketch.count() == 0);

  sketch.update_many({1.0, 2.0, 3.0});
  assert(sketch.count() == 3);
  // The exported sketch carries the same export size through finalize().
  assert(sketch.finalize().size_limit() == 64);
}

/** Random compaction via from_sorted_values (the unit-weight path) is reproducible. */
void test_from_sorted_values_random_reproducible() {
  std::vector<double> values;
  for (int i = 0; i < 100; ++i) {
    values.push_back(static_cast<double>(i));
  }

  const auto a =
      WeightedSketch1D::from_sorted_values(values, 16, CompactMode::Random, 123);
  const auto b =
      WeightedSketch1D::from_sorted_values(values, 16, CompactMode::Random, 123);
  const auto c =
      WeightedSketch1D::from_sorted_values(values, 16, CompactMode::Random, 456);

  const auto av = a.to_arrays();
  const auto bv = b.to_arrays();
  const auto cv = c.to_arrays();

  assert_values_equal(av.values, bv.values);
  assert_weights_equal(av.weights, bv.weights);
  assert(av.values != cv.values || av.weights != cv.weights);

  // Compaction preserves total represented weight regardless of seed.
  assert(a.total_weight() == 100);
  assert(c.total_weight() == 100);
}

}  // namespace

/** Run every test function and print a compact progress log. */
int main() {
  const std::vector<std::pair<std::string, void (*)()>> tests = {
      {"test_weighted_from_samples_exact_sort_combine_pad",
       test_weighted_from_samples_exact_sort_combine_pad},
      {"test_weighted_quantile_expected_values",
       test_weighted_quantile_expected_values},
      {"test_weighted_quantiles_batch_query",
       test_weighted_quantiles_batch_query},
      {"test_even_quantile_uses_inclusive_lower_median",
       test_even_quantile_uses_inclusive_lower_median},
      {"test_from_arrays_sorts_combines_and_ignores_zero_padding",
       test_from_arrays_sorts_combines_and_ignores_zero_padding},
      {"test_weighted_compaction_deterministic_expected",
       test_weighted_compaction_deterministic_expected},
      {"test_odd_compaction_carries_unpaired_item",
       test_odd_compaction_carries_unpaired_item},
      {"test_odd_compaction_drop_largest_legacy_policy",
       test_odd_compaction_drop_largest_legacy_policy},
      {"test_sorted_value_duplicate_compaction_combines_retained_values",
       test_sorted_value_duplicate_compaction_combines_retained_values},
      {"test_weighted_compaction_random_seed_reproducible",
       test_weighted_compaction_random_seed_reproducible},
      {"test_build_sketch_from_column_sorts_defensively",
       test_build_sketch_from_column_sorts_defensively},
      {"test_merge_expected_and_no_mutation", test_merge_expected_and_no_mutation},
      {"test_merge_two_combine_equal_flag", test_merge_two_combine_equal_flag},
      {"test_padding_does_not_affect_median_helper",
       test_padding_does_not_affect_median_helper},
      {"test_kll_1d_basic_and_nan_behavior", test_kll_1d_basic_and_nan_behavior},
      {"test_kll_1d_batch_update_paths", test_kll_1d_batch_update_paths},
      {"test_kll_1d_compaction_weight_powers_of_two",
       test_kll_1d_compaction_weight_powers_of_two},
      {"test_kll_finalize_fixed_size_is_optional_and_preserves_weight",
       test_kll_finalize_fixed_size_is_optional_and_preserves_weight},
      {"test_kll_nd_update_and_finalize", test_kll_nd_update_and_finalize},
      {"test_merge_weighted_sketches_tree_reduction",
       test_merge_weighted_sketches_tree_reduction},
      {"test_merge_weighted_sketches_compacts_once_when_fixed_size",
       test_merge_weighted_sketches_compacts_once_when_fixed_size},
      {"test_build_sketch_from_sorted_values_and_merge_ops",
       test_build_sketch_from_sorted_values_and_merge_ops},
      {"test_rank_error_helpers", test_rank_error_helpers},
      {"test_error_paths", test_error_paths},
      {"test_kll_1d_size_limit_accessor", test_kll_1d_size_limit_accessor},
      {"test_from_sorted_values_random_reproducible",
       test_from_sorted_values_random_reproducible},
  };

  for (const auto& test : tests) {
    test.second();
    std::cout << "[ok] " << test.first << '\n';
  }
  std::cout << "\nPassed " << tests.size() << " tests.\n";
  return 0;
}
