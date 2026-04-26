#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>

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

float measure_copy_ms(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    int iters) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaMemcpy(dst, src, bytes, kind));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, kind));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return elapsed_ms / iters;
}

bool check_equal(const int* a, const int* b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      std::cerr << "mismatch at " << i << ": " << a[i] << " vs " << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int iters = 20;
  constexpr std::array<size_t, 4> k_sizes = {
      1u << 20,
      4u << 20,
      16u << 20,
      64u << 20,
  };
  constexpr size_t max_bytes = 64u << 20;
  constexpr size_t max_n = max_bytes / sizeof(int);

  int* pageable_src = static_cast<int*>(std::malloc(max_bytes));
  int* pageable_dst = static_cast<int*>(std::malloc(max_bytes));
  int* pinned_src = nullptr;
  int* pinned_dst = nullptr;
  int* dev = nullptr;

  if (pageable_src == nullptr || pageable_dst == nullptr) {
    std::cerr << "malloc failed" << std::endl;
    return EXIT_FAILURE;
  }

  CHECK_CUDA(cudaMallocHost(&pinned_src, max_bytes));
  CHECK_CUDA(cudaMallocHost(&pinned_dst, max_bytes));
  CHECK_CUDA(cudaMalloc(&dev, max_bytes));

  for (size_t i = 0; i < max_n; ++i) {
    pageable_src[i] = static_cast<int>(i);
    pinned_src[i] = static_cast<int>(i);
    pageable_dst[i] = -1;
    pinned_dst[i] = -1;
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "iters: " << iters << '\n';
  std::cout << "size_mb pageable_h2d_GBps pageable_d2h_GBps pinned_h2d_GBps pinned_d2h_GBps pageable_check pinned_check"
            << '\n';

  bool all_ok = true;
  for (size_t bytes : k_sizes) {
    const size_t n = bytes / sizeof(int);
    const float pageable_h2d_ms =
        measure_copy_ms(dev, pageable_src, bytes, cudaMemcpyHostToDevice, iters);
    const float pageable_d2h_ms =
        measure_copy_ms(pageable_dst, dev, bytes, cudaMemcpyDeviceToHost, iters);
    const bool pageable_ok = check_equal(pageable_src, pageable_dst, n);

    const float pinned_h2d_ms =
        measure_copy_ms(dev, pinned_src, bytes, cudaMemcpyHostToDevice, iters);
    const float pinned_d2h_ms =
        measure_copy_ms(pinned_dst, dev, bytes, cudaMemcpyDeviceToHost, iters);
    const bool pinned_ok = check_equal(pinned_src, pinned_dst, n);

    const double gb = static_cast<double>(bytes) / 1.0e9;
    const double pageable_h2d_gbps = gb / (pageable_h2d_ms / 1000.0);
    const double pageable_d2h_gbps = gb / (pageable_d2h_ms / 1000.0);
    const double pinned_h2d_gbps = gb / (pinned_h2d_ms / 1000.0);
    const double pinned_d2h_gbps = gb / (pinned_d2h_ms / 1000.0);

    std::cout << (bytes >> 20) << ' ' << pageable_h2d_gbps << ' ' << pageable_d2h_gbps
              << ' ' << pinned_h2d_gbps << ' ' << pinned_d2h_gbps << ' '
              << (pageable_ok ? "PASS" : "FAIL") << ' '
              << (pinned_ok ? "PASS" : "FAIL") << '\n';
    all_ok = all_ok && pageable_ok && pinned_ok;
  }

  CHECK_CUDA(cudaFree(dev));
  CHECK_CUDA(cudaFreeHost(pinned_src));
  CHECK_CUDA(cudaFreeHost(pinned_dst));
  std::free(pageable_src);
  std::free(pageable_dst);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
