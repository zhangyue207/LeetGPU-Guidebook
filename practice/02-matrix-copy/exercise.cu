#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void matrix_copy_kernel(const float *in, float *out, int rows, int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    // TODO(student): copy in[row * cols + col] to out[row * cols + col].
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("02-matrix-copy", "matrix_copy_kernel");
  }

  constexpr int rows = 1025;
  constexpr int cols = 769;
  constexpr int n = rows * cols;
  const dim3 block(16, 16);
  const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

  std::vector<float> in(n), out(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>((i * 17) % 1009) * 0.25f;
  }

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  matrix_copy_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (!check_close(out, in)) {
    std::cerr << "FAIL: matrix copy output mismatch\n";
    return 1;
  }

  std::cout << "PASS: matrix copy\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
