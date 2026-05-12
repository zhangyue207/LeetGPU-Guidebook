#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TODO(student): write c[i] = a[i] + b[i].
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("01-vector-add", "vector_add_kernel");
  }

  constexpr int n = 1 << 20;
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  std::vector<float> a(n), b(n), c(n, 0.0f), expected(n);
  for (int i = 0; i < n; ++i) {
    a[i] = std::sin(0.001f * i);
    b[i] = std::cos(0.002f * i);
    expected[i] = a[i] + b[i];
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  if (!check_close(c, expected)) {
    std::cerr << "FAIL: vector add output mismatch\n";
    return 1;
  }

  std::cout << "PASS: vector add\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
