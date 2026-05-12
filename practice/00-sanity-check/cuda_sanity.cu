#include "cuda_check.h"

#include <iostream>
#include <vector>

__global__ void write_index_kernel(int *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = i;
  }
}

int main() {
  constexpr int n = 1024;
  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  int *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));

  write_index_kernel<<<grid_size, block_size>>>(d_out, n);
  CUDA_CHECK(cudaGetLastError());

  std::vector<int> out(n, -1);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));

  for (int i = 0; i < n; ++i) {
    if (out[i] != i) {
      std::cerr << "FAIL: out[" << i << "]=" << out[i] << " expected=" << i << "\n";
      return 1;
    }
  }

  std::cout << "PASS: CUDA sanity check\n";
  return 0;
}

