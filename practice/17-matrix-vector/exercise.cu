#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <cmath>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = true;

__global__ void matrix_vector_kernel(const float *matrix, const float *x, float *y, int rows, int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int col = threadIdx.x;

  // TODO(student): load matrix[row * cols + col] * x[col] into shared[col] when valid.
  shared[col] = row < gridDim.x && col < blockDim.x ? matrix[row * cols + col] * x[col] : 0;
  __syncthreads();
  // TODO(student): reduce shared[] so thread 0 writes y[row].
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[col] += shared[col + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[blockIdx.x] = shared[0];
  }
  (void)shared;
  (void)row;
  (void)col;
}

void matrix_vector_cpu(const std::vector<float> &matrix,
                       const std::vector<float> &x,
                       std::vector<float> &y,
                       int rows,
                       int cols) {
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += matrix[row * cols + col] * x[col];
    }
    y[row] = sum;
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("17-matrix-vector", "matrix_vector_kernel");
  }

  constexpr int rows = 512;
  constexpr int cols = 256;
  constexpr int matrix_size = rows * cols;
  const int block_size = cols;
  const int grid_size = rows;
  const std::size_t shared_bytes = block_size * sizeof(float);

  std::vector<float> matrix(matrix_size), x(cols), y(rows, 0.0f), expected(rows, 0.0f);
  for (int i = 0; i < matrix_size; ++i) {
    matrix[i] = static_cast<float>((i * 7) % 101) * 0.01f - 0.5f;
  }
  for (int col = 0; col < cols; ++col) {
    x[col] = std::sin(0.01f * col);
  }
  matrix_vector_cpu(matrix, x, expected, rows, cols);

  float *d_matrix = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_matrix, matrix_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  matrix_vector_kernel<<<grid_size, block_size, shared_bytes>>>(d_matrix, d_x, d_y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_matrix));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  if (!check_close(y, expected, 1e-3f, 1e-4f)) {
    std::cerr << "FAIL: matrix-vector output mismatch\n";
    return 1;
  }

  std::cout << "PASS: matrix-vector\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
