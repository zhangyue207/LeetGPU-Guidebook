#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

__device__ __forceinline__ float4 rope_low4(const float4& q_low,
                                            const float4& q_high,
                                            const float4& c_low,
                                            const float4& s_low) {
  return make_float4(
      fmaf(q_low.x, c_low.x, -q_high.x * s_low.x),
      fmaf(q_low.y, c_low.y, -q_high.y * s_low.y),
      fmaf(q_low.z, c_low.z, -q_high.z * s_low.z),
      fmaf(q_low.w, c_low.w, -q_high.w * s_low.w));
}

__device__ __forceinline__ float4 rope_high4(const float4& q_low,
                                             const float4& q_high,
                                             const float4& c_high,
                                             const float4& s_high) {
  return make_float4(
      fmaf(q_high.x, c_high.x, q_low.x * s_high.x),
      fmaf(q_high.y, c_high.y, q_low.y * s_high.y),
      fmaf(q_high.z, c_high.z, q_low.z * s_high.z),
      fmaf(q_high.w, c_high.w, q_low.w * s_high.w));
}

template <int kHalf>
__global__ void rope_pairs_vec4_128(const float* __restrict__ Q,
                                    const float* __restrict__ cos,
                                    const float* __restrict__ sin,
                                    float* __restrict__ output,
                                    int M) {
  constexpr int kWidth = 4;
  constexpr int kD = kHalf * 2;
  constexpr int kVecsPerRow = kHalf / kWidth;

  const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_vecs = M * kVecsPerRow;
  if (vec_idx >= total_vecs) {
    return;
  }

  const int row = vec_idx / kVecsPerRow;
  const int vec_col = vec_idx - row * kVecsPerRow;
  const int base = row * kD + vec_col * kWidth;

  const float4 q_low = reinterpret_cast<const float4*>(Q + base)[0];
  const float4 q_high = reinterpret_cast<const float4*>(Q + base + kHalf)[0];
  const float4 c_low = reinterpret_cast<const float4*>(cos + base)[0];
  const float4 c_high = reinterpret_cast<const float4*>(cos + base + kHalf)[0];
  const float4 s_low = reinterpret_cast<const float4*>(sin + base)[0];
  const float4 s_high = reinterpret_cast<const float4*>(sin + base + kHalf)[0];

  reinterpret_cast<float4*>(output + base)[0] = rope_low4(q_low, q_high, c_low, s_low);
  reinterpret_cast<float4*>(output + base + kHalf)[0] = rope_high4(q_low, q_high, c_high, s_high);
}

__global__ void rope_pairs_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ cos,
                                  const float* __restrict__ sin,
                                  float* __restrict__ output,
                                  int M,
                                  int D) {
  const size_t half = static_cast<size_t>(D >> 1);
  const size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total_pairs = static_cast<size_t>(M) * half;
  if (pair_idx >= total_pairs) {
    return;
  }

  const size_t row = pair_idx / half;
  const size_t col = pair_idx - row * half;
  const size_t base = row * static_cast<size_t>(D) + col;
  const size_t high = base + half;

  const float q_low = Q[base];
  const float q_high = Q[high];
  output[base] = fmaf(q_low, cos[base], -q_high * sin[base]);
  output[high] = fmaf(q_high, cos[high], q_low * sin[high]);
}

}  // namespace

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
  if (M <= 0 || D <= 0) {
    return;
  }

  constexpr int kThreads = 256;
  if (D == 128) {
    constexpr int kHalf = 64;
    constexpr int kVecsPerRow = kHalf / 4;
    const int total_vecs = M * kVecsPerRow;
    const int blocks = (total_vecs + kThreads - 1) / kThreads;
    rope_pairs_vec4_128<kHalf><<<blocks, kThreads>>>(Q, cos, sin, output, M);
    return;
  }

  const size_t total_pairs = static_cast<size_t>(M) * static_cast<size_t>(D >> 1);
  const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
  rope_pairs_kernel<<<blocks, kThreads>>>(Q, cos, sin, output, M, D);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-5f;

std::vector<float> rope_reference(const std::vector<float>& q,
                                  const std::vector<float>& cos,
                                  const std::vector<float>& sin,
                                  int M,
                                  int D) {
  std::vector<float> expected(M * D, 0.0f);
  const int half = D / 2;
  for (int row = 0; row < M; ++row) {
    const int base = row * D;
    for (int col = 0; col < D; ++col) {
      const int pair_col = (col < half) ? (col + half) : (col - half);
      const float rotated = (col < half) ? -q[base + pair_col] : q[base + pair_col];
      expected[base + col] = q[base + col] * cos[base + col] + rotated * sin[base + col];
    }
  }
  return expected;
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kEps;
}

std::vector<float> make_patterned_data(int n, int period, float scale, float bias) {
  std::vector<float> values(n);
  for (int i = 0; i < n; ++i) {
    const int centered = (i % period) - (period / 2);
    values[i] = centered * scale + bias;
  }
  return values;
}

bool run_case(const std::string& name,
              const std::vector<float>& q,
              const std::vector<float>& cos,
              const std::vector<float>& sin,
              int M,
              int D) {
  const size_t bytes = static_cast<size_t>(M) * static_cast<size_t>(D) * sizeof(float);
  const std::vector<float> expected = rope_reference(q, cos, sin, M, D);
  std::vector<float> actual(M * D, 0.0f);

  float* d_q = nullptr;
  float* d_cos = nullptr;
  float* d_sin = nullptr;
  float* d_out = nullptr;

  auto cleanup = [&]() {
    if (d_q != nullptr) cudaFree(d_q);
    if (d_cos != nullptr) cudaFree(d_cos);
    if (d_sin != nullptr) cudaFree(d_sin);
    if (d_out != nullptr) cudaFree(d_out);
  };

  if (cudaMalloc(&d_q, bytes) != cudaSuccess ||
      cudaMalloc(&d_cos, bytes) != cudaSuccess ||
      cudaMalloc(&d_sin, bytes) != cudaSuccess ||
      cudaMalloc(&d_out, bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  bool ok = true;
  ok = ok && (cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_cos, cos.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_sin, sin.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);

  if (!ok) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed\n";
    cleanup();
    return false;
  }

  solve(d_q, d_cos, d_sin, d_out, M, D);
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution or cudaMemcpy D2H failed\n";
    cleanup();
    return false;
  }

  for (int i = 0; i < M * D; ++i) {
    if (!almost_equal(actual[i], expected[i])) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << i
                << ", expected=" << expected[i] << ", actual=" << actual[i] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int M = 2;
    const int D = 4;
    const std::vector<float> q = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<float> cos = {
        1.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    const std::vector<float> sin = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    passed += run_case("sample_case_identity_and_rotation", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int D = 2;
    const std::vector<float> q = {2.5f, -4.0f};
    const std::vector<float> cos = {0.0f, 0.0f};
    const std::vector<float> sin = {1.0f, 1.0f};
    passed += run_case("smallest_even_dimension", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int D = 6;
    const std::vector<float> q = {
        1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f,
    };
    const std::vector<float> cos = {
        0.5f, -1.0f, 0.25f, 2.0f, -0.5f, 1.5f,
    };
    const std::vector<float> sin = {
        1.0f, 0.5f, -1.0f, 0.25f, 2.0f, -0.5f,
    };
    passed += run_case("mixed_signs_and_coefficients", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 3;
    const int D = 4;
    const std::vector<float> q = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, -2.0f, 5.0f, 6.0f,
        0.5f, -0.5f, 1.5f, -1.5f,
    };
    const std::vector<float> cos = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };
    const std::vector<float> sin = {
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        -0.5f, 1.0f, -1.0f, 0.5f,
    };
    passed += run_case("multiple_rows_independent_positions", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 2;
    const int D = 8;
    const std::vector<float> q = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        -8.0f, -7.0f, -6.0f, -5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    };
    const std::vector<float> cos = {
        1.0f, 0.5f, 0.0f, -0.5f, -1.0f, 1.5f, -1.5f, 2.0f,
        2.0f, -2.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f,
    };
    const std::vector<float> sin = {
        -1.0f, 1.0f, -0.5f, 0.5f, 1.0f, -1.0f, 1.5f, -1.5f,
        0.25f, 0.75f, -0.25f, -0.75f, 1.25f, -1.25f, 0.1f, -0.1f,
    };
    passed += run_case("larger_dimension_stress_pattern", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 2;
    const int D = 128;
    const std::vector<float> q = make_patterned_data(M * D, 19, 0.125f, -0.25f);
    const std::vector<float> cos = make_patterned_data(M * D, 23, 0.05f, 0.5f);
    const std::vector<float> sin = make_patterned_data(M * D, 29, -0.04f, -0.1f);
    passed += run_case("specialized_path_d128", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int D = 2048;
    const std::vector<float> q = make_patterned_data(M * D, 31, 0.03125f, 0.75f);
    const std::vector<float> cos = make_patterned_data(M * D, 27, -0.02f, 0.3f);
    const std::vector<float> sin = make_patterned_data(M * D, 25, 0.015f, -0.45f);
    passed += run_case("generic_large_dimension", q, cos, sin, M, D) ? 1 : 0;
  }

  std::cout << "Summary: " << passed << "/" << total << " cases passed\n";
  return passed == total ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
