#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void rope_kernel(const float *x,
                            const float *cos_table,
                            const float *sin_table,
                            float *y,
                            int tokens,
                            int heads,
                            int head_dim) {
  const int pair_dim = head_dim / 2;
  const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_pairs = tokens * heads * pair_dim;

  if (pair_index < total_pairs) {
    // TODO(student): decode pair_index into token, head, and pair_col.
    // TODO(student): rotate the even/odd values and write both outputs.
  }
}

void rope_cpu(const std::vector<float> &x,
              const std::vector<float> &cos_table,
              const std::vector<float> &sin_table,
              std::vector<float> &y,
              int tokens,
              int heads,
              int head_dim) {
  const int pair_dim = head_dim / 2;
  for (int token = 0; token < tokens; ++token) {
    for (int head = 0; head < heads; ++head) {
      for (int pair_col = 0; pair_col < pair_dim; ++pair_col) {
        const int base = (token * heads + head) * head_dim + 2 * pair_col;
        const float even = x[base];
        const float odd = x[base + 1];
        const float c = cos_table[token * pair_dim + pair_col];
        const float s = sin_table[token * pair_dim + pair_col];
        y[base] = even * c - odd * s;
        y[base + 1] = even * s + odd * c;
      }
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("08-rope", "rope_kernel");
  }

  constexpr int tokens = 128;
  constexpr int heads = 8;
  constexpr int head_dim = 64;
  constexpr int pair_dim = head_dim / 2;
  constexpr int n = tokens * heads * head_dim;
  constexpr int total_pairs = tokens * heads * pair_dim;
  constexpr int block_size = 256;
  const int grid_size = (total_pairs + block_size - 1) / block_size;

  std::vector<float> x(n), y(n, 0.0f), expected(n, 0.0f);
  std::vector<float> cos_table(tokens * pair_dim), sin_table(tokens * pair_dim);

  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i * 11) % 211) * 0.015625f - 1.5f;
  }
  for (int token = 0; token < tokens; ++token) {
    for (int pair_col = 0; pair_col < pair_dim; ++pair_col) {
      const float theta = static_cast<float>(token) * (0.01f + 0.001f * static_cast<float>(pair_col));
      cos_table[token * pair_dim + pair_col] = std::cos(theta);
      sin_table[token * pair_dim + pair_col] = std::sin(theta);
    }
  }
  rope_cpu(x, cos_table, sin_table, expected, tokens, heads, head_dim);

  float *d_x = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cos, cos_table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sin, sin_table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cos, cos_table.data(), cos_table.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin, sin_table.data(), sin_table.size() * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  rope_kernel<<<grid_size, block_size>>>(d_x, d_cos, d_sin, d_y, tokens, heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_cos));
  CUDA_CHECK(cudaFree(d_sin));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-5f, 1e-5f)) {
    std::cerr << "FAIL: RoPE output mismatch\n";
    return 1;
  }

  std::cout << "PASS: RoPE\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
