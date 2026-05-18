#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;
constexpr int kTileDim = 16;

#pragma nv_diag_suppress 177
__global__ void tiled_matmul_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
  __shared__ float tile_a[kTileDim][kTileDim];
  __shared__ float tile_b[kTileDim][kTileDim];

  const int row = blockIdx.y * kTileDim + threadIdx.y;
  const int col = blockIdx.x * kTileDim + threadIdx.x;

  // TODO(student): loop over K dimension tiles.
  for (int j = 0; j < (k + kTileDim - 1) / kTileDim; j++) {
    // TODO(student): load A and B tiles into shared memory with boundary checks.
    tile_a[threadIdx.y][threadIdx.x] = a[row * k + j * kTileDim + threadIdx.x];
    tile_b[threadIdx.y][threadIdx.x] = b[(j * kTileDim + threadIdx.y) * n + ];
    __syncthreads();
    // TODO(student): accumulate tile products and write c[row * n + col] when valid.
    
  }
  (void)row;
  (void)col;
}
#pragma nv_diag_default 177

void matmul_cpu(const std::vector<float> &a,
                const std::vector<float> &b,
                std::vector<float> &c,
                int m,
                int n,
                int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += a[row * k + kk] * b[kk * n + col];
      }
      c[row * n + col] = sum;
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("18-tiled-matmul", "tiled_matmul_kernel");
  }

  constexpr int m = 129;
  constexpr int n = 131;
  constexpr int k = 127;
  const dim3 block(kTileDim, kTileDim);
  const dim3 grid((n + kTileDim - 1) / kTileDim, (m + kTileDim - 1) / kTileDim);

  std::vector<float> a(m * k), b(k * n), c(m * n, 0.0f), expected(m * n, 0.0f);
  for (int i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>((i * 3) % 37) * 0.03125f - 0.5f;
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>((i * 5) % 41) * 0.03125f - 0.5f;
  }
  matmul_cpu(a, b, expected, m, n, k);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  tiled_matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  if (!check_close(c, expected, 1e-3f, 1e-3f)) {
    std::cerr << "FAIL: tiled matmul output mismatch\n";
    return 1;
  }

  std::cout << "PASS: tiled matmul\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
