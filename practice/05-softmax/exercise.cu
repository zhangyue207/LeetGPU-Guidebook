#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void softmax_rows_kernel(const float *x, float *y, int rows, int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  const bool valid = row < rows && col < cols;

  // TODO(student): use shared[] to reduce each row to its maximum value.
  // TODO(student): compute expf(x - max) and reduce those values to the row sum.
  // TODO(student): write y[row * cols + col] = exp_value / row_sum for valid elements.
  (void)shared;
  if (valid) {
    y[row * cols + col] = 0.0f;
  }
}

void softmax_cpu(const std::vector<float> &x, std::vector<float> &y, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    float row_max = x[base];
    for (int col = 1; col < cols; ++col) {
      row_max = std::max(row_max, x[base + col]);
    }

    float denom = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float v = std::exp(x[base + col] - row_max);
      y[base + col] = v;
      denom += v;
    }

    for (int col = 0; col < cols; ++col) {
      y[base + col] /= denom;
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("05-softmax", "softmax_rows_kernel");
  }

  constexpr int rows = 128;
  constexpr int cols = 256;
  constexpr int n = rows * cols;
  const int block_size = cols;
  const int grid_size = rows;
  const std::size_t shared_bytes = cols * sizeof(float);

  std::vector<float> x(n), y(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i * 31) % 97) * 0.03125f - 1.5f;
  }
  softmax_cpu(x, expected, rows, cols);

  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  softmax_rows_kernel<<<grid_size, block_size, shared_bytes>>>(d_x, d_y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-5f, 1e-4f)) {
    std::cerr << "FAIL: softmax output mismatch\n";
    return 1;
  }

  std::cout << "PASS: softmax\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
