#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void dot_blocks_kernel(const float *a, const float *b, float *partial_sums, int n) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  (void)global_tid;
  (void)stride;

  // TODO(student): accumulate a[i] * b[i] with a grid-stride loop.
  // TODO(student): reduce one local sum per thread in shared memory.
  if (tid == 0) {
    partial_sums[blockIdx.x] = 0.0f;
  }
}

float dot_cpu(const std::vector<float> &a, const std::vector<float> &b) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("13-vector-dot", "dot_blocks_kernel");
  }

  constexpr int n = (1 << 22) + 123;
  constexpr int block_size = 256;
  constexpr int grid_size = 256;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>((i % 29) - 14) * 0.03125f;
    b[i] = static_cast<float>((i % 17) - 8) * 0.0625f;
  }
  const float expected = dot_cpu(a, b);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  dot_blocks_kernel<<<grid_size, block_size, shared_bytes>>>(d_a, d_b, d_partial, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  std::vector<float> partial(grid_size);
  CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_partial));

  float actual = 0.0f;
  for (float v : partial) {
    actual += v;
  }

  const float tolerance = 1e-3f + 1e-5f * std::fabs(expected);
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL: dot mismatch actual=" << actual << " expected=" << expected << "\n";
    return 1;
  }

  std::cout << "PASS: vector dot\n";
  std::cout << "dot: " << actual << "\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
