#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <algorithm>
#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = false;

__global__ void max_pooling_kernel(const float *input, float *output, int in_height, int in_width) {
  const int out_width = in_width / 2;
  const int out_height = in_height / 2;
  const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_row = blockIdx.y * blockDim.y + threadIdx.y;

  if (out_row < out_height && out_col < out_width) {
    // TODO(student): compute max over the 2x2 input window for this output element.
    output[out_row * out_width + out_col] = 0.0f;
  }
}

void max_pooling_cpu(const std::vector<float> &input,
                     std::vector<float> &output,
                     int in_height,
                     int in_width) {
  const int out_height = in_height / 2;
  const int out_width = in_width / 2;
  for (int out_row = 0; out_row < out_height; ++out_row) {
    for (int out_col = 0; out_col < out_width; ++out_col) {
      const int base_row = out_row * 2;
      const int base_col = out_col * 2;
      float best = input[base_row * in_width + base_col];
      best = std::max(best, input[base_row * in_width + base_col + 1]);
      best = std::max(best, input[(base_row + 1) * in_width + base_col]);
      best = std::max(best, input[(base_row + 1) * in_width + base_col + 1]);
      output[out_row * out_width + out_col] = best;
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("20-max-pooling", "max_pooling_kernel");
  }

  constexpr int in_height = 256;
  constexpr int in_width = 320;
  constexpr int out_height = in_height / 2;
  constexpr int out_width = in_width / 2;
  constexpr int input_size = in_height * in_width;
  constexpr int output_size = out_height * out_width;
  const dim3 block(16, 16);
  const dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);

  std::vector<float> input(input_size), output(output_size, 0.0f), expected(output_size, 0.0f);
  for (int i = 0; i < input_size; ++i) {
    input[i] = static_cast<float>((i * 19) % 257) * 0.01f - 1.0f;
  }
  max_pooling_cpu(input, expected, in_height, in_width);

  float *d_input = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  max_pooling_kernel<<<grid, block>>>(d_input, d_output, in_height, in_width);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  if (!check_close(output, expected)) {
    std::cerr << "FAIL: max pooling output mismatch\n";
    return 1;
  }

  std::cout << "PASS: max pooling\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
