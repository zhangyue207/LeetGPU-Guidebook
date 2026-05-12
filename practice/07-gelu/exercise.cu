#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void gelu_kernel(const float *x, float *y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TODO(student): implement the tanh-approximation GELU formula for x[i].
    y[i] = 0.0f;
  }
}

float gelu_cpu_value(float v) {
  constexpr float kAlpha = 0.7978845608028654f;
  constexpr float kBeta = 0.044715f;
  return 0.5f * v * (1.0f + std::tanh(kAlpha * (v + kBeta * v * v * v)));
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("07-gelu", "gelu_kernel");
  }

  constexpr int n = (1 << 20) + 17;
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  std::vector<float> x(n), y(n, 0.0f), expected(n);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i * 29) % 257) * 0.03125f - 4.0f;
    expected[i] = gelu_cpu_value(x[i]);
  }

  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  gelu_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-5f, 1e-5f)) {
    std::cerr << "FAIL: GELU output mismatch\n";
    return 1;
  }

  std::cout << "PASS: GELU\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
