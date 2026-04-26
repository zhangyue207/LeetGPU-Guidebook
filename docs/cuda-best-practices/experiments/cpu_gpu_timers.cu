#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t status__ = (call);                                            \
    if (status__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__)             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (false)

__global__ void fill_array(int* a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] = idx;
  }
}

}  // namespace

int main() {
  int* a = nullptr;
  int n = 100000000;
  CHECK_CUDA(cudaMalloc(&a, static_cast<size_t>(n) * sizeof(int)));

  cudaEvent_t e_start;
  cudaEvent_t e_end;
  CHECK_CUDA(cudaEventCreate(&e_start));
  CHECK_CUDA(cudaEventCreate(&e_end));

  fill_array<<<(n + 255) / 256, 256>>>(a, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  auto submit_start = std::chrono::high_resolution_clock::now();
  CHECK_CUDA(cudaEventRecord(e_start));
  fill_array<<<(n + 255) / 256, 256>>>(a, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(e_end));
  auto submit_end = std::chrono::high_resolution_clock::now();

  float kernel_ms = 0.0f;
  CHECK_CUDA(cudaEventSynchronize(e_end));
  CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, e_start, e_end));

  auto e2e_start = std::chrono::high_resolution_clock::now();
  fill_array<<<(n + 255) / 256, 256>>>(a, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  auto e2e_end = std::chrono::high_resolution_clock::now();

  const double host_submit_ms =
      std::chrono::duration<double, std::milli>(submit_end - submit_start).count();
  const double end_to_end_ms =
      std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "num_elements: " << n << '\n';
  std::cout << "host_submit_ms: " << host_submit_ms << '\n';
  std::cout << "kernel_ms: " << kernel_ms << '\n';
  std::cout << "end_to_end_ms: " << end_to_end_ms << '\n';

  CHECK_CUDA(cudaEventDestroy(e_start));
  CHECK_CUDA(cudaEventDestroy(e_end));
  CHECK_CUDA(cudaFree(a));
  return 0;
}
