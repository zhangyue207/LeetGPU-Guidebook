#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void histogram_kernel(const int *values, int *histogram, int n, int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // TODO(student): read values[i] and atomicAdd one count into histogram[bin].
    (void)num_bins;
  }
}

void histogram_cpu(const std::vector<int> &values, std::vector<int> &histogram) {
  for (int v : values) {
    ++histogram[v];
  }
}

bool check_equal_ints(const std::vector<int> &actual, const std::vector<int> &expected) {
  if (actual.size() != expected.size()) {
    std::cerr << "Size mismatch\n";
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "Mismatch at bin " << i << ": actual=" << actual[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("14-histogram", "histogram_kernel");
  }

  constexpr int n = 1 << 20;
  constexpr int num_bins = 64;
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  std::vector<int> values(n), histogram(num_bins, 0), expected(num_bins, 0);
  for (int i = 0; i < n; ++i) {
    values[i] = (i * 17 + i / 7) % num_bins;
  }
  histogram_cpu(values, expected);

  int *d_values = nullptr;
  int *d_histogram = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));

  GpuTimer timer;
  timer.start();
  histogram_kernel<<<grid_size, block_size>>>(d_values, d_histogram, n, num_bins);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(histogram.data(), d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_histogram));

  if (!check_equal_ints(histogram, expected)) {
    std::cerr << "FAIL: histogram mismatch\n";
    return 1;
  }

  std::cout << "PASS: histogram\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
