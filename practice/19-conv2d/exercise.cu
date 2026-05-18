#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;
constexpr int kKernelSize = 3;
constexpr int kRadius = 1;

__global__ void conv2d_kernel(const float *input, const float *kernel, float *output, int height, int width) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    // TODO(student): apply the 3x3 kernel around (row, col) with zero padding.
    output[row * width + col] = 0.0f;
  }
}

void conv2d_cpu(const std::vector<float> &input,
                const std::vector<float> &kernel,
                std::vector<float> &output,
                int height,
                int width) {
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      float sum = 0.0f;
      for (int ky = 0; ky < kKernelSize; ++ky) {
        for (int kx = 0; kx < kKernelSize; ++kx) {
          const int in_row = row + ky - kRadius;
          const int in_col = col + kx - kRadius;
          if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
            sum += input[in_row * width + in_col] * kernel[ky * kKernelSize + kx];
          }
        }
      }
      output[row * width + col] = sum;
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("19-conv2d", "conv2d_kernel");
  }

  constexpr int height = 257;
  constexpr int width = 263;
  constexpr int n = height * width;
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  std::vector<float> input(n), kernel(kKernelSize * kKernelSize), output(n, 0.0f), expected(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    input[i] = static_cast<float>((i * 11) % 113) * 0.01f - 0.5f;
  }
  const float kernel_values[kKernelSize * kKernelSize] = {
      0.0f, -1.0f, 0.0f,
      -1.0f, 5.0f, -1.0f,
      0.0f, -1.0f, 0.0f,
  };
  for (int i = 0; i < kKernelSize * kKernelSize; ++i) {
    kernel[i] = kernel_values[i];
  }
  conv2d_cpu(input, kernel, expected, height, width);

  float *d_input = nullptr;
  float *d_kernel = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, kernel.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  conv2d_kernel<<<grid, block>>>(d_input, d_kernel, d_output, height, width);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_output));

  if (!check_close(output, expected, 1e-4f, 1e-4f)) {
    std::cerr << "FAIL: conv2d output mismatch\n";
    return 1;
  }

  std::cout << "PASS: conv2d\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
