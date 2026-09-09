# Contributing to minikll

Thanks for your interest in improving `minikll`. It is a small, dependency-free
header-only library, and the goal is to keep it that way.

## Build and test

The core library is header-only; building is only needed to run the tests.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional components are off by default:

```bash
cmake -S . -B build -DMINIKLL_BUILD_PYTHON=ON      # pybind11 bindings
cmake -S . -B build -DMINIKLL_BUILD_EXAMPLES=ON     # HDF5 ANN-Benchmarks example
```

Please make sure `ctest` passes (and, ideally, that the build is warning-clean
under `-Wall -Wextra`) before opening a pull request.

## Adding a test

Tests live in `tests/test_minikll.cpp` and use plain `assert` (no external
framework). To add one:

1. Write a `void test_my_case()` function using the existing helpers
   (`assert_close`, `assert_values_equal`, `assert_weights_equal`,
   `assert_throws`, ...).
2. Register it in the `tests` table inside `main()`.

Prefer small, deterministic cases. For randomized behavior, pin a seed and assert
reproducibility rather than specific compacted values.

## Code style

- C++17, header-only core with no runtime dependencies. Keep new public code in
  `minikll.hpp`; keep implementation helpers in `namespace detail`.
- 2-space indentation, `lower_snake_case` for functions and variables,
  `CamelCase` for types.
- Document every public type and function with a `/** ... */` comment.
- Validate public inputs and throw `std::invalid_argument` with a clear message,
  matching the existing functions.
- If you change the public API, update `README.md`, the Python bindings in
  `bindings/python/`, and the tests in the same change.

## License

By contributing, you agree that your contributions are licensed under the MIT
License (see `LICENSE`).
