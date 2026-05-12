#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void rmsnorm_rows_kernel(const float *x,
                                    const float *weight,
                                    float *y,
                                    int rows,
                                    int cols,
                                    float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int col = threadIdx.x;

  // TODO(student): load x[row * cols + col]^2 into shared[col] when col < cols, otherwise 0.
  // TODO(student): reduce shared[] to get the row sum of squares.
  // TODO(student): compute inv_rms = rsqrtf(sum_squares / cols + eps).
  // TODO(student): write y[row * cols + col] = x[...] * inv_rms * weight[col] when col < cols.
  (void)shared;
  if (row < rows && col < cols) {
    y[row * cols + col] = 0.0f;
  }
}

void rmsnorm_cpu(const std::vector<float> &x,
                 const std::vector<float> &weight,
                 std::vector<float> &y,
                 int rows,
                 int cols,
                 float eps) {
  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    float sum_squares = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum_squares += x[base + col] * x[base + col];
    }

    const float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(cols) + eps);
    for (int col = 0; col < cols; ++col) {
      y[base + col] = x[base + col] * inv_rms * weight[col];
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("06-rmsnorm", "rmsnorm_rows_kernel");
  }

  constexpr int rows = 128;
  constexpr int cols = 256;
  constexpr int n = rows * cols;
  constexpr float eps = 1e-5f;
  const int block_size = 256;
  const int grid_size = rows;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> x(n), weight(cols), y(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i * 19) % 127) * 0.03125f - 2.0f;
  }
  for (int col = 0; col < cols; ++col) {
    weight[col] = 0.75f + static_cast<float>((col * 7) % 17) * 0.015625f;
  }
  rmsnorm_cpu(x, weight, expected, rows, cols, eps);

  float *d_x = nullptr;
  float *d_weight = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  rmsnorm_rows_kernel<<<grid_size, block_size, shared_bytes>>>(d_x, d_weight, d_y, rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-4f, 1e-4f)) {
    std::cerr << "FAIL: RMSNorm output mismatch\n";
    return 1;
  }

  std::cout << "PASS: RMSNorm\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
