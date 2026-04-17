#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kItemsPerThread = 8;
constexpr int kTileItems = kBlockThreads * kItemsPerThread;

template <typename T>
__device__ __forceinline__ int merge_path_partition(const T* a,
                                                    int a_count,
                                                    const T* b,
                                                    int b_count,
                                                    int diag) {
  int low = max(0, diag - b_count);
  int high = min(diag, a_count);

  while (low < high) {
    const int mid = (low + high) >> 1;
    const int j = diag - mid;

    if (j > 0 && mid < a_count && b[j - 1] >= a[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

__global__ void parallel_merge_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      int m,
                                      int n) {
  __shared__ float tile[kTileItems];

  const int total = m + n;
  const int block_diag = blockIdx.x * kTileItems;
  if (block_diag >= total) {
    return;
  }

  const int tile_count = min(kTileItems, total - block_diag);
  const int block_a_start = merge_path_partition(a, m, b, n, block_diag);
  const int block_a_end = merge_path_partition(a, m, b, n, block_diag + tile_count);
  const int block_b_start = block_diag - block_a_start;
  const int block_b_end = block_diag + tile_count - block_a_end;
  const int tile_a_count = block_a_end - block_a_start;
  const int tile_b_count = block_b_end - block_b_start;

  float* tile_a = tile;
  float* tile_b = tile + tile_a_count;

  for (int idx = threadIdx.x; idx < tile_a_count; idx += blockDim.x) {
    tile_a[idx] = a[block_a_start + idx];
  }
  for (int idx = threadIdx.x; idx < tile_b_count; idx += blockDim.x) {
    tile_b[idx] = b[block_b_start + idx];
  }
  __syncthreads();

  const int thread_diag = min(threadIdx.x * kItemsPerThread, tile_count);
  const int thread_diag_end = min(thread_diag + kItemsPerThread, tile_count);

  const int thread_a_start =
      merge_path_partition(tile_a, tile_a_count, tile_b, tile_b_count, thread_diag);
  const int thread_a_end =
      merge_path_partition(tile_a, tile_a_count, tile_b, tile_b_count, thread_diag_end);
  const int thread_b_start = thread_diag - thread_a_start;
  const int thread_b_end = thread_diag_end - thread_a_end;

  int a_idx = thread_a_start;
  int b_idx = thread_b_start;
  float out[kItemsPerThread];

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const bool active = thread_diag + item < thread_diag_end;
    const bool take_a =
        active &&
        (b_idx >= thread_b_end || (a_idx < thread_a_end && tile_a[a_idx] <= tile_b[b_idx]));
    out[item] = take_a ? tile_a[a_idx++] : (active ? tile_b[b_idx++] : 0.0f);
  }

  const int out_base = block_diag + thread_diag;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    if (thread_diag + item < tile_count) {
      c[out_base + item] = out[item];
    }
  }
}

}  // namespace

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  const int total = M + N;
  if (total == 0) {
    return;
  }

  const int blocks = (total + kTileItems - 1) / kTileItems;
  parallel_merge_kernel<<<blocks, kBlockThreads>>>(A, B, C, M, N);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-6f;
constexpr float kSentinel = 123456.0f;

std::vector<float> merge_reference(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> expected;
  expected.reserve(a.size() + b.size());

  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      expected.push_back(a[i++]);
    } else {
      expected.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    expected.push_back(a[i++]);
  }
  while (j < b.size()) {
    expected.push_back(b[j++]);
  }
  return expected;
}

std::vector<float> make_sorted_data(int count, float start, float step, int wobble_mod) {
  std::vector<float> values(count);
  for (int idx = 0; idx < count; ++idx) {
    values[idx] = start + step * static_cast<float>(idx) +
                  0.001f * static_cast<float>(idx % wobble_mod);
  }
  return values;
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kEps;
}

bool run_case(const std::string& name,
              const std::vector<float>& a,
              const std::vector<float>& b) {
  const int m = static_cast<int>(a.size());
  const int n = static_cast<int>(b.size());
  const size_t bytes_a = static_cast<size_t>(m) * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(n) * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(m + n) * sizeof(float);

  const std::vector<float> expected = merge_reference(a, b);
  std::vector<float> actual(m + n, 0.0f);
  const std::vector<float> sentinel(m + n, kSentinel);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  auto cleanup = [&]() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
  };

  if (cudaMalloc(&d_a, bytes_a) != cudaSuccess ||
      cudaMalloc(&d_b, bytes_b) != cudaSuccess ||
      cudaMalloc(&d_c, bytes_c) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  bool ok = true;
  ok = ok && (cudaMemcpy(d_a, a.data(), bytes_a, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_b, b.data(), bytes_b, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_c, sentinel.data(), bytes_c, cudaMemcpyHostToDevice) == cudaSuccess);
  if (!ok) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed\n";
    cleanup();
    return false;
  }

  solve(d_a, d_b, d_c, m, n);
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_c, bytes_c, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution or cudaMemcpy D2H failed\n";
    cleanup();
    return false;
  }

  bool unchanged = true;
  for (float value : actual) {
    if (!almost_equal(value, kSentinel)) {
      unchanged = false;
      break;
    }
  }
  if (unchanged) {
    std::cerr << "[FAIL] " << name << ": output buffer was never updated by solve\n";
    cleanup();
    return false;
  }

  for (int idx = 0; idx < m + n; ++idx) {
    if (!almost_equal(actual[idx], expected[idx])) {
      std::cerr << "[FAIL] " << name << ": mismatch at index " << idx
                << ", expected=" << expected[idx] << ", actual=" << actual[idx] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

bool run_large_case() {
  const std::vector<float> a = make_sorted_data(1 << 18, -1000.0f, 0.5f, 7);
  const std::vector<float> b = make_sorted_data(1 << 18, -999.75f, 0.5f, 11);
  return run_case("large_balanced_interleaved_case", a, b);
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const std::vector<float> a = {1.0f, 3.0f, 5.0f, 7.0f};
    const std::vector<float> b = {2.0f, 4.0f, 6.0f, 8.0f};
    passed += run_case("sample_case_interleaved_even_lengths", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-1.0f, 1.0f, 3.0f};
    const std::vector<float> b = {2.0f};
    passed += run_case("sample_case_single_tail_insert", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {1.5f};
    const std::vector<float> b = {-2.5f};
    passed += run_case("minimum_sizes_one_element_each", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-4.0f, -1.0f, 2.0f, 2.0f};
    const std::vector<float> b = {-3.0f, 2.0f, 5.0f};
    passed += run_case("duplicates_and_negative_values", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-10.0f, -9.0f, -8.0f};
    const std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    passed += run_case("one_array_entirely_before_the_other", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {0.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> b = {0.0f, 0.0f, 2.0f};
    passed += run_case("repeated_equal_values", a, b) ? 1 : 0;
  }

  {
    ++total;
    passed += run_large_case() ? 1 : 0;
  }

  std::cout << "Summary: " << passed << "/" << total << " cases passed\n";
  return passed == total ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
