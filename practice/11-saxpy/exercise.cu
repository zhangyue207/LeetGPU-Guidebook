#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void saxpy_kernel(float alpha, const float *x, float *y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TODO(student): write y[i] = alpha * x[i] + y[i].
    y[i] = alpha * x[i] + y[i];
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("11-saxpy", "saxpy_kernel");
  }

  constexpr int n = (1 << 20) + 19;
  constexpr float alpha = 1.75f;
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  std::vector<float> x(n), y(n), expected(n);
  for (int i = 0; i < n; ++i) {
    x[i] = std::sin(0.001f * i);
    y[i] = std::cos(0.002f * i);
    expected[i] = alpha * x[i] + y[i];
  }

  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  saxpy_kernel<<<grid_size, block_size>>>(alpha, d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected)) {
    std::cerr << "FAIL: SAXPY output mismatch\n";
    return 1;
  }

  std::cout << "PASS: SAXPY\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
