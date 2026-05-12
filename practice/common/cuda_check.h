#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

inline void cuda_check(cudaError_t status, const char *expr, const char *file, int line) {
  if (status == cudaSuccess) {
    return;
  }

  std::cerr << "CUDA error at " << file << ":" << line << "\n"
            << "  expression: " << expr << "\n"
            << "  name: " << cudaGetErrorName(status) << "\n"
            << "  message: " << cudaGetErrorString(status) << "\n";
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

