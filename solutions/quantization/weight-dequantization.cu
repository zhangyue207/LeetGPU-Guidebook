#include <cuda_runtime.h>

#ifndef ONLINE_JUDGE
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#endif

namespace {

constexpr int kBaselineBlockEdge = 16;
constexpr int kScalarBlockSize = 256;
constexpr int kVectorBlockSize = 256;

int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

template <int kTileSize>
__global__ void weight_dequant_vec4_kernel(const float* X, const float* S, float* Y, int M, int N) {
  const int row = blockIdx.y;
  const int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = vec_index * 4;
  if (row >= M || col >= N) {
    return;
  }

  constexpr int kTileShift = kTileSize == 16 ? 4 : kTileSize == 32 ? 5 : kTileSize == 64 ? 6 : 7;
  const int scale_cols = (N + kTileSize - 1) >> kTileShift;
  const float scale = S[(row >> kTileShift) * scale_cols + (col >> kTileShift)];

  const float4 values = reinterpret_cast<const float4*>(X + row * N + col)[0];
  float4 result;
  result.x = values.x * scale;
  result.y = values.y * scale;
  result.z = values.z * scale;
  result.w = values.w * scale;
  reinterpret_cast<float4*>(Y + row * N + col)[0] = result;
}

template <int kTileSize>
__global__ void weight_dequant_tile_vec4_kernel(const float* X, const float* S, float* Y, int M, int N) {
  constexpr int kBlockRows = 8;
  constexpr int kVecWidth = 4;
  constexpr int kTileShift = kTileSize == 16 ? 4 : kTileSize == 32 ? 5 : kTileSize == 64 ? 6 : 7;

  const int tile_col = blockIdx.x;
  const int row = blockIdx.y * kBlockRows + threadIdx.y;
  const int col = tile_col * kTileSize + threadIdx.x * kVecWidth;
  if (row >= M || col >= N) {
    return;
  }

  const int scale_cols = (N + kTileSize - 1) >> kTileShift;
  const float scale = S[(row >> kTileShift) * scale_cols + tile_col];
  const float4 values = reinterpret_cast<const float4*>(X + row * N + col)[0];
  float4 result;
  result.x = values.x * scale;
  result.y = values.y * scale;
  result.z = values.z * scale;
  result.w = values.w * scale;
  reinterpret_cast<float4*>(Y + row * N + col)[0] = result;
}

template <int kTileSize>
__global__ void weight_dequant_vec4x2_kernel(const float* X, const float* S, float* Y, int M, int N) {
  const int row = blockIdx.y;
  const int vec_index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const int col0 = vec_index * 4;
  const int col1 = col0 + 4;
  if (row >= M || col0 >= N) {
    return;
  }

  constexpr int kTileShift = kTileSize == 16 ? 4 : kTileSize == 32 ? 5 : kTileSize == 64 ? 6 : 7;
  const int scale_cols = (N + kTileSize - 1) >> kTileShift;
  const int scale_row = row >> kTileShift;

  const float scale0 = S[scale_row * scale_cols + (col0 >> kTileShift)];
  const float4 values0 = reinterpret_cast<const float4*>(X + row * N + col0)[0];
  float4 result0;
  result0.x = values0.x * scale0;
  result0.y = values0.y * scale0;
  result0.z = values0.z * scale0;
  result0.w = values0.w * scale0;
  reinterpret_cast<float4*>(Y + row * N + col0)[0] = result0;

  if (col1 < N) {
    const float scale1 = S[scale_row * scale_cols + (col1 >> kTileShift)];
    const float4 values1 = reinterpret_cast<const float4*>(X + row * N + col1)[0];
    float4 result1;
    result1.x = values1.x * scale1;
    result1.y = values1.y * scale1;
    result1.z = values1.z * scale1;
    result1.w = values1.w * scale1;
    reinterpret_cast<float4*>(Y + row * N + col1)[0] = result1;
  }
}

__global__ void weight_dequant_scalar_kernel(const float* X,
                                             const float* S,
                                             float* Y,
                                             int M,
                                             int N,
                                             int tile_size) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  const int scale_cols = (N + tile_size - 1) / tile_size;
  Y[row * N + col] = X[row * N + col] * S[(row / tile_size) * scale_cols + (col / tile_size)];
}

__global__ void weight_dequant_baseline_kernel(const float* X,
                                               const float* S,
                                               float* Y,
                                               int M,
                                               int N,
                                               int tile_size) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bxo = blockIdx.x * tile_size;
  const int byo = blockIdx.y * tile_size;
  const int scale_cols = (N + tile_size - 1) / tile_size;

  for (int yoffset = 0; yoffset < tile_size; yoffset += kBaselineBlockEdge) {
    for (int xoffset = 0; xoffset < tile_size; xoffset += kBaselineBlockEdge) {
      const int xo = xoffset + bxo + tx;
      const int yo = yoffset + byo + ty;
      if (xo < N && yo < M) {
        Y[yo * N + xo] = X[yo * N + xo] * S[(yo / tile_size) * scale_cols + (xo / tile_size)];
      }
    }
  }
}

void solve_baseline_impl(const float* X, const float* S, float* Y, int M, int N, int tile_size) {
  weight_dequant_baseline_kernel<<<dim3(ceil_div(N, tile_size), ceil_div(M, tile_size)),
                                   dim3(kBaselineBlockEdge, kBaselineBlockEdge)>>>(
      X, S, Y, M, N, tile_size);
}

void solve_candidate_impl(const float* X, const float* S, float* Y, int M, int N, int tile_size) {
  if ((N & 3) == 0 && tile_size >= 64) {
    switch (tile_size) {
      case 64:
        weight_dequant_tile_vec4_kernel<64><<<dim3(ceil_div(N, 64), ceil_div(M, 8)), dim3(16, 8)>>>(
            X, S, Y, M, N);
        return;
      case 128:
        weight_dequant_tile_vec4_kernel<128><<<dim3(ceil_div(N, 128), ceil_div(M, 8)),
                                               dim3(32, 8)>>>(X, S, Y, M, N);
        return;
      default:
        break;
    }
  }

  if ((N & 3) == 0) {
    const dim3 block(kVectorBlockSize);
    const dim3 grid(ceil_div(N / 4, kVectorBlockSize), M);
    switch (tile_size) {
      case 16:
        weight_dequant_vec4_kernel<16><<<grid, block>>>(X, S, Y, M, N);
        return;
      case 32:
        weight_dequant_vec4_kernel<32><<<grid, block>>>(X, S, Y, M, N);
        return;
      case 64:
        weight_dequant_vec4_kernel<64><<<grid, block>>>(X, S, Y, M, N);
        return;
      case 128:
        weight_dequant_vec4_kernel<128><<<grid, block>>>(X, S, Y, M, N);
        return;
      default:
        break;
    }
  }

  weight_dequant_scalar_kernel<<<dim3(ceil_div(N, kScalarBlockSize), M), kScalarBlockSize>>>(
      X, S, Y, M, N, tile_size);
}

void solve_candidate_variant_impl(const float* X,
                                  const float* S,
                                  float* Y,
                                  int M,
                                  int N,
                                  int tile_size,
                                  int variant_id) {
  switch (variant_id) {
    case 0:
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    case 1:
      if ((N & 3) == 0) {
        const dim3 block(128);
        const dim3 grid(ceil_div(N / 8, 128), M);
        switch (tile_size) {
          case 16:
            weight_dequant_vec4x2_kernel<16><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 32:
            weight_dequant_vec4x2_kernel<32><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 64:
            weight_dequant_vec4x2_kernel<64><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 128:
            weight_dequant_vec4x2_kernel<128><<<grid, block>>>(X, S, Y, M, N);
            return;
          default:
            break;
        }
      }
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    case 2:
      if ((N & 3) == 0) {
        const dim3 block(128);
        const dim3 grid(ceil_div(N / 4, 128), M);
        switch (tile_size) {
          case 16:
            weight_dequant_vec4_kernel<16><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 32:
            weight_dequant_vec4_kernel<32><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 64:
            weight_dequant_vec4_kernel<64><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 128:
            weight_dequant_vec4_kernel<128><<<grid, block>>>(X, S, Y, M, N);
            return;
          default:
            break;
        }
      }
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    case 3:
      if ((N & 3) == 0) {
        switch (tile_size) {
          case 16:
            weight_dequant_tile_vec4_kernel<16><<<dim3(ceil_div(N, 16), ceil_div(M, 8)),
                                                  dim3(4, 8)>>>(X, S, Y, M, N);
            return;
          case 32:
            weight_dequant_tile_vec4_kernel<32><<<dim3(ceil_div(N, 32), ceil_div(M, 8)),
                                                  dim3(8, 8)>>>(X, S, Y, M, N);
            return;
          case 64:
            weight_dequant_tile_vec4_kernel<64><<<dim3(ceil_div(N, 64), ceil_div(M, 8)),
                                                  dim3(16, 8)>>>(X, S, Y, M, N);
            return;
          case 128:
            weight_dequant_tile_vec4_kernel<128><<<dim3(ceil_div(N, 128), ceil_div(M, 8)),
                                                   dim3(32, 8)>>>(X, S, Y, M, N);
            return;
          default:
            break;
        }
      }
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    default:
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
  }
}

}  // namespace

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
  solve_candidate_impl(X, S, Y, M, N, TILE_SIZE);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-5f;

std::vector<float> dequant_reference(const std::vector<float>& X,
                                     const std::vector<float>& S,
                                     int M,
                                     int N,
                                     int tile_size) {
  const int scale_cols = ceil_div(N, tile_size);
  std::vector<float> Y(M * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      const int scale_row = i / tile_size;
      const int scale_col = j / tile_size;
      Y[i * N + j] = X[i * N + j] * S[scale_row * scale_cols + scale_col];
    }
  }
  return Y;
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kEps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

bool run_case(const std::string& name,
              const std::vector<float>& X,
              const std::vector<float>& S,
              int M,
              int N,
              int tile_size) {
  const size_t matrix_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  const size_t scale_bytes =
      static_cast<size_t>(ceil_div(M, tile_size)) * static_cast<size_t>(ceil_div(N, tile_size)) *
      sizeof(float);
  const std::vector<float> expected = dequant_reference(X, S, M, N, tile_size);
  std::vector<float> actual(M * N, 0.0f);

  float* d_X = nullptr;
  float* d_S = nullptr;
  float* d_Y = nullptr;
  auto cleanup = [&]() {
    if (d_X != nullptr) cudaFree(d_X);
    if (d_S != nullptr) cudaFree(d_S);
    if (d_Y != nullptr) cudaFree(d_Y);
  };

  if (cudaMalloc(&d_X, matrix_bytes) != cudaSuccess ||
      cudaMalloc(&d_S, scale_bytes) != cudaSuccess ||
      cudaMalloc(&d_Y, matrix_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_X, X.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_S, S.data(), scale_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemset(d_Y, 0, matrix_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": device upload failed\n";
    cleanup();
    return false;
  }

  solve(d_X, d_S, d_Y, M, N, tile_size);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_Y, matrix_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution failed\n";
    cleanup();
    return false;
  }

  for (int index = 0; index < M * N; ++index) {
    if (!almost_equal(actual[index], expected[index])) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << index
                << ", expected=" << expected[index] << ", actual=" << actual[index] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

float time_impl(bool use_candidate,
                const float* d_X,
                const float* d_S,
                float* d_Y,
                int M,
                int N,
                int tile_size,
                int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    if (use_candidate) {
      solve_candidate_impl(d_X, d_S, d_Y, M, N, tile_size);
    } else {
      solve_baseline_impl(d_X, d_S, d_Y, M, N, tile_size);
    }
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    if (use_candidate) {
      solve_candidate_impl(d_X, d_S, d_Y, M, N, tile_size);
    } else {
      solve_baseline_impl(d_X, d_S, d_Y, M, N, tile_size);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iterations);
}

float time_variant(int variant_id,
                   const float* d_X,
                   const float* d_S,
                   float* d_Y,
                   int M,
                   int N,
                   int tile_size,
                   int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    solve_candidate_variant_impl(d_X, d_S, d_Y, M, N, tile_size, variant_id);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    solve_candidate_variant_impl(d_X, d_S, d_Y, M, N, tile_size, variant_id);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iterations);
}

bool run_benchmark_case(int M, int N, int tile_size) {
  const size_t matrix_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  const int scale_rows = ceil_div(M, tile_size);
  const int scale_cols = ceil_div(N, tile_size);
  const size_t scale_bytes = static_cast<size_t>(scale_rows) * static_cast<size_t>(scale_cols) *
                             sizeof(float);

  std::vector<float> X(M * N);
  std::vector<float> S(scale_rows * scale_cols);
  for (int i = 0; i < M * N; ++i) {
    X[i] = static_cast<float>((i % 127) - 63) * 0.125f;
  }
  for (int i = 0; i < scale_rows * scale_cols; ++i) {
    S[i] = 0.25f + static_cast<float>(i % 11) * 0.125f;
  }

  float* d_X = nullptr;
  float* d_S = nullptr;
  float* d_Y = nullptr;
  auto cleanup = [&]() {
    if (d_X != nullptr) cudaFree(d_X);
    if (d_S != nullptr) cudaFree(d_S);
    if (d_Y != nullptr) cudaFree(d_Y);
  };

  if (cudaMalloc(&d_X, matrix_bytes) != cudaSuccess ||
      cudaMalloc(&d_S, scale_bytes) != cudaSuccess ||
      cudaMalloc(&d_Y, matrix_bytes) != cudaSuccess) {
    std::cerr << "[BENCH] allocation failed for M=" << M << ", N=" << N
              << ", TILE_SIZE=" << tile_size << '\n';
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_X, X.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_S, S.data(), scale_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[BENCH] device upload failed for M=" << M << ", N=" << N
              << ", TILE_SIZE=" << tile_size << '\n';
    cleanup();
    return false;
  }

  const int iterations = M >= 4096 ? 50 : 200;
  const float baseline_ms = time_impl(false, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float optimized_ms = time_impl(true, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float variant_vec4x2_ms = time_variant(1, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float variant_vec4_128_ms = time_variant(2, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float variant_tile_ms = time_variant(3, d_X, d_S, d_Y, M, N, tile_size, iterations);
  std::cout << "[BENCH] M=" << std::setw(5) << M
            << " N=" << std::setw(5) << N
            << " TILE_SIZE=" << std::setw(3) << tile_size
            << " baseline=" << std::fixed << std::setprecision(4) << baseline_ms << " ms"
            << " optimized=" << optimized_ms << " ms"
            << " speedup=" << baseline_ms / optimized_ms << "x\n";
  std::cout << "        variants: vec4x1-256=" << optimized_ms
            << " vec4x2-128=" << variant_vec4x2_ms
            << " vec4x1-128=" << variant_vec4_128_ms
            << " tile-vec4=" << variant_tile_ms << '\n';

  cleanup();
  return true;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int M = 4;
    const int N = 4;
    const int tile_size = 2;
    const std::vector<float> X = {
        10.0f, 10.0f, 5.0f, 5.0f,
        10.0f, 10.0f, 5.0f, 5.0f,
        2.0f,  2.0f,  8.0f, 8.0f,
        2.0f,  2.0f,  8.0f, 8.0f,
    };
    const std::vector<float> S = {
        0.5f, 2.0f,
        4.0f, 0.25f,
    };
    passed += run_case("sample_case_prompt_matrix_tiles", X, S, M, N, tile_size) ? 1 : 0;
  }

  {
    ++total;
    const int M = 3;
    const int N = 5;
    const int tile_size = 2;
    const std::vector<float> X = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    };
    const std::vector<float> S = {
        1.0f, 0.5f, -1.0f,
        2.0f, -0.25f, 4.0f,
    };
    passed += run_case("partial_tiles_non_multiple_dimensions", X, S, M, N, tile_size) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 3;
    const int tile_size = 16;
    const std::vector<float> X = {3.0f, -2.0f, 7.5f};
    const std::vector<float> S = {0.25f};
    passed += run_case("single_row_tile_larger_than_matrix", X, S, M, N, tile_size) ? 1 : 0;
  }

  {
    ++total;
    const int M = 5;
    const int N = 4;
    const int tile_size = 4;
    const std::vector<float> X = {
        -1.0f, -2.0f, -3.0f, -4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        -9.0f, 10.0f, -11.0f, 12.0f,
        13.0f, -14.0f, 15.0f, -16.0f,
        2.0f,  4.0f,  6.0f,  8.0f,
    };
    const std::vector<float> S = {
        -0.5f,
        2.0f,
    };
    passed += run_case("mixed_signs_across_row_tiles", X, S, M, N, tile_size) ? 1 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
  if (passed == total) {
    run_benchmark_case(1024, 1024, 16);
    run_benchmark_case(2048, 2048, 32);
    run_benchmark_case(4096, 4096, 64);
    run_benchmark_case(8192, 8192, 128);
  }
  return passed == total ? 0 : 1;
}
#endif
