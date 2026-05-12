#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = true;

__global__ void sum_blocks_kernel(const float *x, float *partial_sums, int n) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)i;

  // TODO(student): load x[i] into shared[tid] when i < n, otherwise 0.
  shared[tid] = i < n ? x[i] : 0;
  __syncthreads();

  // TODO(student): reduce shared[] so thread 0 writes the block sum to partial_sums[blockIdx.x].
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    partial_sums[blockIdx.x] = shared[0];
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
    return todo_exit("04-reduction", "sum_blocks_kernel");
  }

  constexpr int n = (1 << 20) + 123;
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> x(n);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i % 23) - 11) * 0.125f;
  }
  const float expected = sum_cpu(x);

  float *d_x = nullptr;
  float *d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  sum_blocks_kernel<<<grid_size, block_size, shared_bytes>>>(d_x, d_partial, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  std::vector<float> partial(grid_size);
  CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_partial));

  const float actual = sum_cpu(partial);
  const float tolerance = 1e-3f + 1e-5f * std::fabs(expected);
  if (std::fabs(actual - expected) > tolerance) {
    std::cerr << "FAIL: reduction mismatch actual=" << actual << " expected=" << expected << "\n";
    return 1;
  }

  std::cout << "PASS: reduction\n";
  std::cout << "sum: " << actual << "\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
