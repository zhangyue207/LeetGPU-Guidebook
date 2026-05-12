#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void matrix_transpose_kernel(const float *in, float *out, int rows, int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    // TODO(student): write out[col * rows + row] = in[row * cols + col].
  }
}

void transpose_cpu(const std::vector<float> &in, std::vector<float> &out, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      out[col * rows + row] = in[row * cols + col];
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("03-matrix-transpose", "matrix_transpose_kernel");
  }

  constexpr int rows = 1024;
  constexpr int cols = 768;
  constexpr int n = rows * cols;
  const dim3 block(16, 16);
  const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

  std::vector<float> in(n), out(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>((i * 13) % 2048) * 0.125f;
  }
  transpose_cpu(in, expected, rows, cols);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  matrix_transpose_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (!check_close(out, expected)) {
    std::cerr << "FAIL: matrix transpose output mismatch\n";
    return 1;
  }

  std::cout << "PASS: matrix transpose\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
