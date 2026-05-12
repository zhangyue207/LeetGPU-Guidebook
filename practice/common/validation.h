#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

inline bool nearly_equal(float actual, float expected, float atol = 1e-5f, float rtol = 1e-5f) {
  const float diff = std::fabs(actual - expected);
  return diff <= atol + rtol * std::fabs(expected);
}

inline bool check_close(const std::vector<float> &actual,
                        const std::vector<float> &expected,
                        float atol = 1e-5f,
                        float rtol = 1e-5f,
                        int max_report = 8) {
  if (actual.size() != expected.size()) {
    std::cerr << "Size mismatch: actual=" << actual.size() << " expected=" << expected.size() << "\n";
    return false;
  }

  int mismatches = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!nearly_equal(actual[i], expected[i], atol, rtol)) {
      if (mismatches < max_report) {
        std::cerr << "Mismatch at " << i << ": actual=" << actual[i]
                  << " expected=" << expected[i] << "\n";
      }
      ++mismatches;
    }
  }

  if (mismatches != 0) {
    std::cerr << "Total mismatches: " << mismatches << "\n";
    return false;
  }

  return true;
}

inline int todo_exit(const char *exercise_name, const char *kernel_name) {
  std::cout << "[" << exercise_name << "] TODO\n"
            << "  1. Open this exercise.cu file.\n"
            << "  2. Implement " << kernel_name << ".\n"
            << "  3. Set kStudentKernelImplemented = true.\n"
            << "  4. Rebuild and run this executable again.\n";
  return 2;
}

