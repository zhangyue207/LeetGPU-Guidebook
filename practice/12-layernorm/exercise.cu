#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = true;

__global__ void layernorm_rows_kernel(const float *x,
                                      const float *gamma,
                                      const float *beta,
                                      float *y,
                                      int rows,
                                      int cols,
                                      float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int col = threadIdx.x;

  // TODO(student): reduce x[row, :] to compute the row mean.
  shared[col] = col < cols ? x[row * cols + col]: 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (col < s) {
      shared[col] += shared[col + s]; 
    }
    __syncthreads();
  }
  float mean_ = shared[0] / cols;
  // TODO(student): reduce squared deviations to compute variance.
  shared[col] = col < cols ? (x[row * cols + col] - mean_)* (x[row * cols + col] - mean_) : 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (col < s) {
      shared[col] += shared[col + s]; 
    }
    __syncthreads();
  }
  float var_ = shared[0] / cols;
  // TODO(student): write normalized output with gamma[col] and beta[col].
  (void)shared;
  if (row < rows && col < cols) {
    y[row * cols + col] = (x[row * cols + col] - mean_) * rsqrtf(var_ + eps) * gamma[col] + beta[col];
  }
}

void layernorm_cpu(const std::vector<float> &x,
                   const std::vector<float> &gamma,
                   const std::vector<float> &beta,
                   std::vector<float> &y,
                   int rows,
                   int cols,
                   float eps) {
  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += x[base + col];
    }
    const float mean = sum / static_cast<float>(cols);

    float sum_sq = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float centered = x[base + col] - mean;
      sum_sq += centered * centered;
    }
    const float inv_std = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);

    for (int col = 0; col < cols; ++col) {
      y[base + col] = (x[base + col] - mean) * inv_std * gamma[col] + beta[col];
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("12-layernorm", "layernorm_rows_kernel");
  }

  constexpr int rows = 128;
  constexpr int cols = 256;
  constexpr int n = rows * cols;
  constexpr float eps = 1e-5f;
  const int block_size = cols;
  const int grid_size = rows;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> x(n), gamma(cols), beta(cols), y(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i * 37) % 251) * 0.015625f - 2.0f;
  }
  for (int col = 0; col < cols; ++col) {
    gamma[col] = 0.8f + static_cast<float>((col * 5) % 23) * 0.01f;
    beta[col] = static_cast<float>((col * 3) % 11) * 0.02f - 0.1f;
  }
  layernorm_cpu(x, gamma, beta, expected, rows, cols, eps);

  float *d_x = nullptr;
  float *d_gamma = nullptr;
  float *d_beta = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  layernorm_rows_kernel<<<grid_size, block_size, shared_bytes>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-4f, 1e-4f)) {
    std::cerr << "FAIL: LayerNorm output mismatch\n";
    return 1;
  }

  std::cout << "PASS: LayerNorm\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
