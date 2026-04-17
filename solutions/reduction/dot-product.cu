#include <cuda_runtime.h>

#ifndef ONLINE_JUDGE
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#endif

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void dot_product_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ result,
                                   int N) {
  float thread_sum = 0.0f;
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = global_thread; i < N; i += stride) {
    thread_sum = fmaf(A[i], B[i], thread_sum);
  }

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  thread_sum = warp_reduce_sum(thread_sum);

  if (lane == 0) {
    warp_sums[warp] = thread_sum;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    float block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      atomicAdd(result, block_sum);
    }
  }
}

}  // namespace

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
  if (N <= 0) {
    return;
  }

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  int blocks = (N + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }

  cudaMemset(result, 0, sizeof(float));
  dot_product_kernel<<<blocks, kThreads>>>(A, B, result, N);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kAbsEps = 1e-5f;
constexpr float kRelEps = 1e-4f;

float reference_dot_product(const std::vector<float>& a, const std::vector<float>& b) {
  double sum = 0.0;
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(sum);
}

bool almost_equal(float actual, float expected) {
  const float diff = std::fabs(actual - expected);
  if (diff <= kAbsEps) {
    return true;
  }
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return diff <= scale * kRelEps;
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
              const std::vector<float>& a,
              const std::vector<float>& b) {
  const int n = static_cast<int>(a.size());
  const size_t vector_bytes = static_cast<size_t>(n) * sizeof(float);
  const float expected = reference_dot_product(a, b);
  float actual = 0.0f;

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_result = nullptr;

  auto cleanup = [&]() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_result != nullptr) cudaFree(d_result);
  };

  if (cudaMalloc(&d_a, vector_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed for A\n";
    cleanup();
    return false;
  }
  if (cudaMalloc(&d_b, vector_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed for B\n";
    cleanup();
    return false;
  }
  if (cudaMalloc(&d_result, sizeof(float)) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed for result\n";
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_a, a.data(), vector_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed for A\n";
    cleanup();
    return false;
  }
  if (cudaMemcpy(d_b, b.data(), vector_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed for B\n";
    cleanup();
    return false;
  }

  const float poison = 12345.0f;
  if (cudaMemcpy(d_result, &poison, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed for result init\n";
    cleanup();
    return false;
  }

  solve(d_a, d_b, d_result, n);

  if (cudaGetLastError() != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel launch failed\n";
    cleanup();
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaDeviceSynchronize failed\n";
    cleanup();
    return false;
  }
  if (cudaMemcpy(&actual, d_result, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy D2H failed for result\n";
    cleanup();
    return false;
  }

  if (!almost_equal(actual, expected)) {
    std::cerr << "[FAIL] " << name << ": expected=" << expected
              << ", actual=" << actual << '\n';
    cleanup();
    return false;
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
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    passed += run_case("example_1_basic_positive_values", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {0.5f, 1.5f, 2.5f};
    const std::vector<float> b = {2.0f, 3.0f, 4.0f};
    passed += run_case("example_2_fractional_inputs", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-3.5f};
    const std::vector<float> b = {2.0f};
    passed += run_case("single_element_minimum_n", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-2.0f, 4.0f, -1.5f, 3.0f, 0.25f};
    const std::vector<float> b = {8.0f, -0.5f, 6.0f, 1.5f, -4.0f};
    passed += run_case("mixed_signs_and_cancellation", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = make_patterned_data(4097, 29, 0.125f, -1.0f);
    const std::vector<float> b = make_patterned_data(4097, 17, -0.25f, 0.5f);
    passed += run_case("larger_deterministic_reference_case", a, b) ? 1 : 0;
  }

  std::cout << passed << "/" << total << " cases passed\n";
  return passed == total ? 0 : 1;
}
#endif
