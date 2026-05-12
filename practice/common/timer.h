#pragma once

#include "cuda_check.h"

class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() {
    CUDA_CHECK(cudaEventRecord(start_));
  }

  float stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return elapsed_ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

