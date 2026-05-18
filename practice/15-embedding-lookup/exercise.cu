#include "cuda_check.h"
#include "timer.h"
#include "validation.h"

#include <iostream>
#include <vector>

constexpr bool kStudentKernelImplemented = true;

__global__ void embedding_lookup_kernel(const float *table,
                                        const int *token_ids,
                                        float *output,
                                        int num_tokens,
                                        int embedding_dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_tokens * embedding_dim;

  if (i < total) {
    // TODO(student): decode i into token and dim.
    int token = i / embedding_dim;
    int dim = i % embedding_dim;
    // TODO(student): copy table[token_ids[token] * embedding_dim + dim] to output[i].
    output[token * embedding_dim + dim] = table[token_ids[token] * embedding_dim + dim];
  }
}

void embedding_lookup_cpu(const std::vector<float> &table,
                          const std::vector<int> &token_ids,
                          std::vector<float> &output,
                          int embedding_dim) {
  for (std::size_t token = 0; token < token_ids.size(); ++token) {
    const int src_base = token_ids[token] * embedding_dim;
    const int dst_base = static_cast<int>(token) * embedding_dim;
    for (int dim = 0; dim < embedding_dim; ++dim) {
      output[dst_base + dim] = table[src_base + dim];
    }
  }
}

int main() {
  if (!kStudentKernelImplemented) {
    return todo_exit("15-embedding-lookup", "embedding_lookup_kernel");
  }

  constexpr int vocab_size = 4096;
  constexpr int num_tokens = 1024;
  constexpr int embedding_dim = 128;
  constexpr int table_size = vocab_size * embedding_dim;
  constexpr int output_size = num_tokens * embedding_dim;
  constexpr int block_size = 256;
  const int grid_size = (output_size + block_size - 1) / block_size;

  std::vector<float> table(table_size), output(output_size, 0.0f), expected(output_size, 0.0f);
  std::vector<int> token_ids(num_tokens);
  for (int i = 0; i < table_size; ++i) {
    table[i] = static_cast<float>((i * 13) % 1021) * 0.001f - 0.5f;
  }
  for (int token = 0; token < num_tokens; ++token) {
    token_ids[token] = (token * 97 + 13) % vocab_size;
  }
  embedding_lookup_cpu(table, token_ids, expected, embedding_dim);

  float *d_table = nullptr;
  float *d_output = nullptr;
  int *d_token_ids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_table, table_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_ids, num_tokens * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_table, table.data(), table_size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_ids, token_ids.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.start();
  embedding_lookup_kernel<<<grid_size, block_size>>>(d_table, d_token_ids, d_output, num_tokens, embedding_dim);
  CUDA_CHECK(cudaGetLastError());
  const float kernel_ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_token_ids));
  CUDA_CHECK(cudaFree(d_output));

  if (!check_close(output, expected)) {
    std::cerr << "FAIL: embedding lookup output mismatch\n";
    return 1;
  }

  std::cout << "PASS: embedding lookup\n";
  std::cout << "kernel_ms: " << kernel_ms << "\n";
  return 0;
}
