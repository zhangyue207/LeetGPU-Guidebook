#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void optimized_sum_blocks_kernel(const float *x, float *partial_sums, int n) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  (void)global_tid;
  (void)stride;

  // TODO(student): use a grid-stride loop to accumulate multiple x[i] values into a local sum.
  // TODO(student): store the local sum in shared[tid].
  // TODO(student): reduce shared[] so thread 0 writes partial_sums[blockIdx.x].
  if (tid == 0) {
    partial_sums[blockIdx.x] = 0.0f;
  }
}

float sum_cpu(const std::vector<float> &x) {
  float sum = 0.0f;
  for (float v : x) {
    sum += v;
  }
  return sum;
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("09-optimized-reduction", "optimized_sum_blocks_kernel");
  }

  constexpr int n = (1 << 22) + 111;
  constexpr int block_size = 256;
  constexpr int grid_size = 256;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> x(n);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i % 31) - 15) * 0.0625f;
  }
  const float expected = sum_cpu(x);

  float *d_x = nullptr;
  float *d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  optimized_sum_blocks_kernel<<<grid_size, block_size, shared_bytes>>>(d_x, d_partial, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  std::vector<float> partial(grid_size);
  CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_partial));

  const float actual = sum_cpu(partial);
  const float tolerance = 1e-3f + 1e-5f * std::fabs(expected);
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL: optimized reduction mismatch actual=" << actual
              << " expected=" << expected << "\n";
    return 1;
  }

  std::cout << "PASS: optimized reduction\n";
  std::cout << "sum: " << actual << "\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
