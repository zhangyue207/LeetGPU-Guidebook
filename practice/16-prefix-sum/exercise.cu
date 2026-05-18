#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void prefix_sum_kernel(const float *x, float *y, int n) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;

  // TODO(student): load x[tid] into shared[tid] when tid < n, otherwise 0.
  shared[tid] = tid < n ? x[tid] : 0;
  __syncthreads();
  // TODO(student): perform an inclusive scan in shared memory.
  float sum = 0;
  for(int i = 0; i <= tid; i++) {
    sum += shared[i];
  }
  // TODO(student): write shared[tid] to y[tid] when tid < n.
  (void)shared;
  (void)tid;
}

void prefix_sum_cpu(const std::vector<float> &x, std::vector<float> &y) {
  float running = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i) {
    running += x[i];
    y[i] = running;
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("16-prefix-sum", "prefix_sum_kernel");
  }

  constexpr int n = 1024;
  constexpr int block_size = 1024;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> x(n), y(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>((i % 13) - 6) * 0.125f;
  }
  prefix_sum_cpu(x, expected);

  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  prefix_sum_kernel<<<1, block_size, shared_bytes>>>(d_x, d_y, n);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-4f, 1e-4f)) {
    std::cerr << "FAIL: prefix sum output mismatch\n";
    return 1;
  }

  std::cout << "PASS: prefix sum\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
