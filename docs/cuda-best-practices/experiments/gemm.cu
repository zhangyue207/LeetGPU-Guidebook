#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

constexpr int k_tile_dim = 32;
constexpr int k_ab_repeats = 50;
constexpr int k_aat_repeats = 50;
constexpr int k_matrix_dim = 1024;
constexpr int k_validation_dim = 128;

__global__ void init_pattern_kernel(float* data, int rows, int cols, float scale) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    int idx = row * cols + col;
    int pattern = (row * 17 + col * 13) % 97;
    data[idx] = scale * static_cast<float>(pattern - 48) / 17.0f;
  }
}

__global__ void gemm_ab_baseline_kernel(const float* a, const float* b, float* c, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < n; ++k) {
    sum += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = sum;
}

__global__ void gemm_ab_tiled_a_kernel(const float* a, const float* b, float* c, int n) {
  __shared__ float a_tile[k_tile_dim][k_tile_dim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + tx;
  int row = blockIdx.y * blockDim.y + ty;

  float sum = 0.0f;
  for (int tile = 0; tile < n; tile += k_tile_dim) {
    if (row < n && tile + tx < n) {
      a_tile[ty][tx] = a[row * n + tile + tx];
    } else {
      a_tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    if (row < n && col < n) {
      for (int k = 0; k < k_tile_dim; ++k) {
        sum += a_tile[ty][k] * b[(tile + k) * n + col];
      }
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}

__global__ void gemm_ab_tiled_ab_kernel(const float* a, const float* b, float* c, int n) {
  __shared__ float a_tile[k_tile_dim][k_tile_dim];
  __shared__ float b_tile[k_tile_dim][k_tile_dim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + tx;
  int row = blockIdx.y * blockDim.y + ty;

  float sum = 0.0f;
  for (int tile = 0; tile < n; tile += k_tile_dim) {
    if (row < n && tile + tx < n) {
      a_tile[ty][tx] = a[row * n + tile + tx];
    } else {
      a_tile[ty][tx] = 0.0f;
    }

    if (tile + ty < n && col < n) {
      b_tile[ty][tx] = b[(tile + ty) * n + col];
    } else {
      b_tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    if (row < n && col < n) {
      for (int k = 0; k < k_tile_dim; ++k) {
        sum += a_tile[ty][k] * b_tile[k][tx];
      }
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}

__global__ void gemm_aat_baseline_kernel(const float* a, float* c, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= rows) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < cols; ++k) {
    sum += a[row * cols + k] * a[col * cols + k];
  }
  c[row * rows + col] = sum;
}

__global__ void gemm_aat_tiled_kernel(const float* a, float* c, int rows, int cols) {
  __shared__ float row_tile[k_tile_dim][k_tile_dim];
  __shared__ float col_tile[k_tile_dim][k_tile_dim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + tx;
  int out_row = blockIdx.y * blockDim.y + ty;

  float sum = 0.0f;
  for (int tile = 0; tile < cols; tile += k_tile_dim) {
    if (out_row < rows && tile + tx < cols) {
      row_tile[ty][tx] = a[out_row * cols + tile + tx];
    } else {
      row_tile[ty][tx] = 0.0f;
    }

    if (out_col < rows && tile + ty < cols) {
      col_tile[tx][ty] = a[out_col * cols + tile + ty];
    } else {
      col_tile[tx][ty] = 0.0f;
    }
    __syncthreads();

    if (out_row < rows && out_col < rows) {
      for (int k = 0; k < k_tile_dim; ++k) {
        sum += row_tile[ty][k] * col_tile[tx][k];
      }
    }
    __syncthreads();
  }

  if (out_row < rows && out_col < rows) {
    c[out_row * rows + out_col] = sum;
  }
}

__global__ void gemm_aat_tiled_padded_kernel(const float* a, float* c, int rows, int cols) {
  __shared__ float row_tile[k_tile_dim][k_tile_dim];
  __shared__ float col_tile[k_tile_dim][k_tile_dim + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + tx;
  int out_row = blockIdx.y * blockDim.y + ty;

  float sum = 0.0f;
  for (int tile = 0; tile < cols; tile += k_tile_dim) {
    if (out_row < rows && tile + tx < cols) {
      row_tile[ty][tx] = a[out_row * cols + tile + tx];
    } else {
      row_tile[ty][tx] = 0.0f;
    }

    if (out_col < rows && tile + ty < cols) {
      col_tile[tx][ty] = a[out_col * cols + tile + ty];
    } else {
      col_tile[tx][ty] = 0.0f;
    }
    __syncthreads();

    if (out_row < rows && out_col < rows) {
      for (int k = 0; k < k_tile_dim; ++k) {
        sum += row_tile[ty][k] * col_tile[tx][k];
      }
    }
    __syncthreads();
  }

  if (out_row < rows && out_col < rows) {
    c[out_row * rows + out_col] = sum;
  }
}

void fill_pattern(float* device_ptr, int rows, int cols, float scale) {
  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  init_pattern_kernel<<<grid, block>>>(device_ptr, rows, cols, scale);
  CHECK_CUDA(cudaGetLastError());
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

void cpu_gemm_ab(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>* c, int n) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
      }
      (*c)[row * n + col] = sum;
    }
  }
}

void cpu_gemm_aat(const std::vector<float>& a, std::vector<float>* c, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < rows; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < cols; ++k) {
        sum += a[row * cols + k] * a[col * cols + k];
      }
      (*c)[row * rows + col] = sum;
    }
  }
}

template <typename LaunchFn>
float measure_ms(int repeats, LaunchFn&& launch) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return elapsed_ms / repeats;
}

void print_device_info() {
  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "gpu: " << prop.name << '\n';
  std::cout << "cuda_capability: " << prop.major << "." << prop.minor << '\n';
}

struct RunResult {
  std::string name;
  float ms;
  double metric;
  float max_error;
  bool pass;
};

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(3);
  print_device_info();
  std::cout << "tile_dim: " << k_tile_dim << '\n';

  const size_t matrix_elems = static_cast<size_t>(k_matrix_dim) * k_matrix_dim;
  const size_t matrix_bytes = matrix_elems * sizeof(float);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, matrix_bytes));
  CHECK_CUDA(cudaMalloc(&d_b, matrix_bytes));
  CHECK_CUDA(cudaMalloc(&d_c, matrix_bytes));

  fill_pattern(d_a, k_matrix_dim, k_matrix_dim, 1.0f);
  fill_pattern(d_b, k_matrix_dim, k_matrix_dim, 0.75f);
  CHECK_CUDA(cudaDeviceSynchronize());

  const int validation_n = k_validation_dim;
  const size_t validation_elems = static_cast<size_t>(validation_n) * validation_n;

  std::vector<float> h_a(validation_elems);
  std::vector<float> h_b(validation_elems);
  std::vector<float> h_ref(validation_elems);
  std::vector<float> h_out(validation_elems);
  float* d_val_a = nullptr;
  float* d_val_b = nullptr;
  float* d_val_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_val_a, validation_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_val_b, validation_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_val_c, validation_elems * sizeof(float)));
  fill_pattern(d_val_a, validation_n, validation_n, 1.0f);
  fill_pattern(d_val_b, validation_n, validation_n, 0.75f);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_a.data(), d_val_a, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(
      h_b.data(), d_val_b, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  cpu_gemm_ab(h_a, h_b, &h_ref, validation_n);

  dim3 block_ab(k_tile_dim, k_tile_dim);
  dim3 grid_ab((k_matrix_dim + k_tile_dim - 1) / k_tile_dim, (k_matrix_dim + k_tile_dim - 1) / k_tile_dim);

  gemm_ab_baseline_kernel<<<grid_ab, block_ab>>>(d_a, d_b, d_c, k_matrix_dim);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<RunResult> ab_results;
  const auto ab_baseline_ms = measure_ms(k_ab_repeats, [&] {
    gemm_ab_baseline_kernel<<<grid_ab, block_ab>>>(d_a, d_b, d_c, k_matrix_dim);
  });
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  gemm_ab_baseline_kernel<<<dim3((validation_n + k_tile_dim - 1) / k_tile_dim,
                                 (validation_n + k_tile_dim - 1) / k_tile_dim),
                            block_ab>>>(d_val_a, d_val_b, d_val_c, validation_n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_out.data(), d_val_c, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  ab_results.push_back({
      "baseline",
      ab_baseline_ms,
      2.0 * static_cast<double>(k_matrix_dim) * k_matrix_dim * k_matrix_dim / (ab_baseline_ms * 1.0e6),
      max_abs_diff(h_ref, h_out),
      max_abs_diff(h_ref, h_out) < 1e-3f});

  const auto ab_tiled_a_ms = measure_ms(k_ab_repeats, [&] {
    gemm_ab_tiled_a_kernel<<<grid_ab, block_ab>>>(d_a, d_b, d_c, k_matrix_dim);
  });
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  gemm_ab_tiled_a_kernel<<<dim3((validation_n + k_tile_dim - 1) / k_tile_dim,
                                (validation_n + k_tile_dim - 1) / k_tile_dim),
                           block_ab>>>(d_val_a, d_val_b, d_val_c, validation_n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_out.data(), d_val_c, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  ab_results.push_back({
      "shared_a",
      ab_tiled_a_ms,
      2.0 * static_cast<double>(k_matrix_dim) * k_matrix_dim * k_matrix_dim / (ab_tiled_a_ms * 1.0e6),
      max_abs_diff(h_ref, h_out),
      max_abs_diff(h_ref, h_out) < 1e-3f});

  const auto ab_tiled_ab_ms = measure_ms(k_ab_repeats, [&] {
    gemm_ab_tiled_ab_kernel<<<grid_ab, block_ab>>>(d_a, d_b, d_c, k_matrix_dim);
  });
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  gemm_ab_tiled_ab_kernel<<<dim3((validation_n + k_tile_dim - 1) / k_tile_dim,
                                 (validation_n + k_tile_dim - 1) / k_tile_dim),
                            block_ab>>>(d_val_a, d_val_b, d_val_c, validation_n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_out.data(), d_val_c, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  ab_results.push_back({
      "shared_ab",
      ab_tiled_ab_ms,
      2.0 * static_cast<double>(k_matrix_dim) * k_matrix_dim * k_matrix_dim / (ab_tiled_ab_ms * 1.0e6),
      max_abs_diff(h_ref, h_out),
      max_abs_diff(h_ref, h_out) < 1e-3f});

  std::cout << "experiment: C=AB" << '\n';
  std::cout << "n: " << k_matrix_dim << '\n';
  std::cout << "repeats: " << k_ab_repeats << '\n';
  std::cout << "variant ms gflops max_error check speedup_vs_baseline" << '\n';
  for (const auto& result : ab_results) {
    std::cout << result.name << ' ' << result.ms << ' ' << result.metric << ' '
              << result.max_error << ' ' << (result.pass ? "PASS" : "FAIL") << ' '
              << (ab_results.front().ms / result.ms) << '\n';
  }

  const int aat_rows = k_matrix_dim;
  const int aat_cols = k_matrix_dim;
  const size_t aat_out_elems = static_cast<size_t>(aat_rows) * aat_rows;
  const size_t aat_out_bytes = aat_out_elems * sizeof(float);

  float* d_aat = nullptr;
  float* d_aat_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_aat, matrix_bytes));
  CHECK_CUDA(cudaMalloc(&d_aat_out, aat_out_bytes));
  fill_pattern(d_aat, aat_rows, aat_cols, 1.25f);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> h_aat(static_cast<size_t>(validation_n) * validation_n);
  std::vector<float> h_aat_ref(validation_elems);
  std::vector<float> h_aat_out(validation_elems);
  float* d_val_aat = nullptr;
  float* d_val_aat_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_val_aat, validation_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_val_aat_out, validation_elems * sizeof(float)));
  fill_pattern(d_val_aat, validation_n, validation_n, 1.25f);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_aat.data(), d_val_aat, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  cpu_gemm_aat(h_aat, &h_aat_ref, validation_n, validation_n);

  dim3 block_aat(k_tile_dim, k_tile_dim);
  dim3 grid_aat((aat_rows + k_tile_dim - 1) / k_tile_dim, (aat_rows + k_tile_dim - 1) / k_tile_dim);

  gemm_aat_baseline_kernel<<<grid_aat, block_aat>>>(d_aat, d_aat_out, aat_rows, aat_cols);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<RunResult> aat_results;
  const auto aat_baseline_ms = measure_ms(k_aat_repeats, [&] {
    gemm_aat_baseline_kernel<<<grid_aat, block_aat>>>(d_aat, d_aat_out, aat_rows, aat_cols);
  });
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  gemm_aat_baseline_kernel<<<dim3((validation_n + k_tile_dim - 1) / k_tile_dim,
                                  (validation_n + k_tile_dim - 1) / k_tile_dim),
                             block_aat>>>(d_val_aat, d_val_aat_out, validation_n, validation_n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_aat_out.data(), d_val_aat_out, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  aat_results.push_back({
      "baseline",
      aat_baseline_ms,
      static_cast<double>(2ull * aat_rows * aat_rows * aat_cols * sizeof(float)) / (aat_baseline_ms * 1.0e6),
      max_abs_diff(h_aat_ref, h_aat_out),
      max_abs_diff(h_aat_ref, h_aat_out) < 1e-3f});

  const auto aat_tiled_ms = measure_ms(k_aat_repeats, [&] {
    gemm_aat_tiled_kernel<<<grid_aat, block_aat>>>(d_aat, d_aat_out, aat_rows, aat_cols);
  });
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  gemm_aat_tiled_kernel<<<dim3((validation_n + k_tile_dim - 1) / k_tile_dim,
                               (validation_n + k_tile_dim - 1) / k_tile_dim),
                          block_aat>>>(d_val_aat, d_val_aat_out, validation_n, validation_n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_aat_out.data(), d_val_aat_out, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  aat_results.push_back({
      "shared",
      aat_tiled_ms,
      static_cast<double>(2ull * aat_rows * aat_rows * aat_cols * sizeof(float)) / (aat_tiled_ms * 1.0e6),
      max_abs_diff(h_aat_ref, h_aat_out),
      max_abs_diff(h_aat_ref, h_aat_out) < 1e-3f});

  const auto aat_padded_ms = measure_ms(k_aat_repeats, [&] {
    gemm_aat_tiled_padded_kernel<<<grid_aat, block_aat>>>(d_aat, d_aat_out, aat_rows, aat_cols);
  });
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  gemm_aat_tiled_padded_kernel<<<dim3((validation_n + k_tile_dim - 1) / k_tile_dim,
                                      (validation_n + k_tile_dim - 1) / k_tile_dim),
                                 block_aat>>>(d_val_aat, d_val_aat_out, validation_n, validation_n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      h_aat_out.data(), d_val_aat_out, validation_elems * sizeof(float), cudaMemcpyDeviceToHost));
  aat_results.push_back({
      "shared_padded",
      aat_padded_ms,
      static_cast<double>(2ull * aat_rows * aat_rows * aat_cols * sizeof(float)) / (aat_padded_ms * 1.0e6),
      max_abs_diff(h_aat_ref, h_aat_out),
      max_abs_diff(h_aat_ref, h_aat_out) < 1e-3f});

  std::cout << "experiment: C=AAT" << '\n';
  std::cout << "rows: " << aat_rows << '\n';
  std::cout << "cols: " << aat_cols << '\n';
  std::cout << "repeats: " << k_aat_repeats << '\n';
  std::cout << "variant ms effective_GBps max_error check speedup_vs_baseline" << '\n';
  for (const auto& result : aat_results) {
    std::cout << result.name << ' ' << result.ms << ' ' << result.metric << ' '
              << result.max_error << ' ' << (result.pass ? "PASS" : "FAIL") << ' '
              << (aat_results.front().ms / result.ms) << '\n';
  }

  bool all_ok = true;
  for (const auto& result : ab_results) {
    all_ok = all_ok && result.pass;
  }
  for (const auto& result : aat_results) {
    all_ok = all_ok && result.pass;
  }

  CHECK_CUDA(cudaFree(d_aat_out));
  CHECK_CUDA(cudaFree(d_aat));
  CHECK_CUDA(cudaFree(d_val_aat_out));
  CHECK_CUDA(cudaFree(d_val_aat));
  CHECK_CUDA(cudaFree(d_val_c));
  CHECK_CUDA(cudaFree(d_val_b));
  CHECK_CUDA(cudaFree(d_val_a));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
