#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;
constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 550
__global__ void tiled_transpose_kernel(const float *in, float *out, int rows, int cols) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;

  // TODO(student): cooperatively load a 32x32 tile from in to shared memory.
  // Use a for loop with step kBlockRows so each thread loads multiple rows.
  (void)x;
  (void)y;

  __syncthreads();

  const int transposed_x = blockIdx.y * kTileDim + threadIdx.x;
  const int transposed_y = blockIdx.x * kTileDim + threadIdx.y;

  // TODO(student): write the transposed tile from shared memory to out.
  // Remember that output shape is cols x rows, so the output stride is rows.
  (void)transposed_x;
  (void)transposed_y;
}
#pragma nv_diag_default 550
#pragma nv_diag_default 177

void transpose_cpu(const std::vector<float> &in, std::vector<float> &out, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      out[col * rows + row] = in[row * cols + col];
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("10-tiled-transpose", "tiled_transpose_kernel");
  }

  constexpr int rows = 1025;
  constexpr int cols = 769;
  constexpr int n = rows * cols;
  const dim3 block(kTileDim, kBlockRows);
  const dim3 grid((cols + kTileDim - 1) / kTileDim, (rows + kTileDim - 1) / kTileDim);

  std::vector<float> in(n), out(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = static_cast<float>((i * 23) % 4096) * 0.0625f;
  }
  transpose_cpu(in, expected, rows, cols);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  tiled_transpose_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (!check_close(out, expected)) {
    std::cerr << "FAIL: tiled transpose output mismatch\n";
    return 1;
  }

  std::cout << "PASS: tiled transpose\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
