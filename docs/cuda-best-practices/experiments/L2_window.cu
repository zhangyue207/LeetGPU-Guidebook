#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
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

constexpr int k_block_size = 256;
constexpr int k_kernel_repeats = 10;
constexpr size_t k_total_region_bytes = 1024u << 20;
constexpr size_t k_desired_set_aside_bytes = 30u << 20;
constexpr size_t k_tuned_window_bytes = 20u << 20;
constexpr int k_freq_sizes_mb[] = {10, 20, 30, 40, 50, 60};

__global__ void init_arr(int* v, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    v[idx] = idx % 1000;
  }
}

__global__ void sliding_window_kernel(
    int* data_persistent,
    int* data_streaming,
    int data_size,
    int freq_size,
    unsigned int* sink) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < data_size) {
    int persistent_idx = tid % freq_size;
    int streaming_idx = tid % data_size;
    int p = data_persistent[persistent_idx];
    int s = data_streaming[streaming_idx];
    p = p * 2 + 1;
    s = s * 2 + 1;
    data_persistent[persistent_idx] = p;
    data_streaming[streaming_idx] = s;
    sink[tid] = static_cast<unsigned int>(p ^ s);
  }
}

void set_access_window(
    cudaStream_t stream,
    void* base_ptr,
    size_t num_bytes,
    float hit_ratio,
    cudaAccessProperty hit_prop,
    cudaAccessProperty miss_prop) {
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = base_ptr;
  attr.accessPolicyWindow.num_bytes = num_bytes;
  attr.accessPolicyWindow.hitRatio = hit_ratio;
  attr.accessPolicyWindow.hitProp = hit_prop;
  attr.accessPolicyWindow.missProp = miss_prop;
  CHECK_CUDA(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

void clear_access_window(cudaStream_t stream) {
  set_access_window(
      stream,
      nullptr,
      0,
      0.0f,
      cudaAccessPropertyNormal,
      cudaAccessPropertyNormal);
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());
}

float measure_case_ms(
    cudaStream_t stream,
    int* data_region,
    unsigned int* sink,
    int total_elems,
    int freq_elems) {
  int* data_persistent = data_region;
  int* data_streaming = data_region + freq_elems;
  const int streaming_elems = total_elems - freq_elems;
  const int window_elems = freq_elems;
  const int num_windows = streaming_elems / window_elems;
  const int blocks = (window_elems + k_block_size - 1) / k_block_size;

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int repeat = 0; repeat < k_kernel_repeats; ++repeat) {
    for (int w = 0; w < num_windows; ++w) {
      sliding_window_kernel<<<blocks, k_block_size, 0, stream>>>(
          data_persistent,
          data_streaming + w * window_elems,
          window_elems,
          freq_elems,
          sink);
    }
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return elapsed_ms / k_kernel_repeats;
}

bool validate_sink(const std::vector<unsigned int>& host_sink) {
  for (unsigned int value : host_sink) {
    if (value != 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  const size_t set_aside_bytes =
      std::min(k_desired_set_aside_bytes, static_cast<size_t>(prop.persistingL2CacheMaxSize));
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside_bytes));

  const int total_elems = static_cast<int>(k_total_region_bytes / sizeof(int));
  int* data_region = nullptr;
  unsigned int* sink = nullptr;
  CHECK_CUDA(cudaMalloc(&data_region, k_total_region_bytes));
  const int max_freq_elems =
      ((k_freq_sizes_mb[sizeof(k_freq_sizes_mb) / sizeof(k_freq_sizes_mb[0]) - 1]) << 20) /
      sizeof(int);
  CHECK_CUDA(cudaMalloc(&sink, static_cast<size_t>(max_freq_elems) * sizeof(unsigned int)));

  init_arr<<<(total_elems + k_block_size - 1) / k_block_size, k_block_size>>>(data_region, total_elems);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::vector<unsigned int> host_sink(static_cast<size_t>(max_freq_elems), 0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "gpu: " << prop.name << '\n';
  std::cout << "l2_cache_mb: " << (prop.l2CacheSize / (1024.0 * 1024.0)) << '\n';
  std::cout << "desired_set_aside_mb: " << (k_desired_set_aside_bytes / (1024.0 * 1024.0)) << '\n';
  std::cout << "persisting_l2_set_aside_mb: " << (set_aside_bytes / (1024.0 * 1024.0)) << '\n';
  std::cout << "access_policy_max_window_mb: "
            << (prop.accessPolicyMaxWindowSize / (1024.0 * 1024.0)) << '\n';
  std::cout << "kernel_repeats: " << k_kernel_repeats << '\n';
  std::cout << "freq_mb streaming_mb baseline_ms fixed_hitratio_ms tuned_hitratio_ms fixed_speedup tuned_speedup sink_check"
            << '\n';

  bool all_ok = true;
  for (int freq_mb : k_freq_sizes_mb) {
    const size_t freq_bytes = static_cast<size_t>(freq_mb) << 20;
    const int freq_elems = static_cast<int>(freq_bytes / sizeof(int));
    const int streaming_mb = static_cast<int>(k_total_region_bytes >> 20) - freq_mb;

    init_arr<<<(total_elems + k_block_size - 1) / k_block_size, k_block_size>>>(data_region, total_elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    clear_access_window(stream);
    const float baseline_ms = measure_case_ms(stream, data_region, sink, total_elems, freq_elems);

    init_arr<<<(total_elems + k_block_size - 1) / k_block_size, k_block_size>>>(data_region, total_elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    const size_t fixed_bytes =
        std::min(freq_bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
    set_access_window(
        stream,
        data_region,
        fixed_bytes,
        1.0f,
        cudaAccessPropertyPersisting,
        cudaAccessPropertyStreaming);
    const float fixed_ms = measure_case_ms(stream, data_region, sink, total_elems, freq_elems);

    init_arr<<<(total_elems + k_block_size - 1) / k_block_size, k_block_size>>>(data_region, total_elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    const size_t tuned_bytes =
        std::min(k_tuned_window_bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
    const float tuned_hit_ratio =
        std::min(1.0f, static_cast<float>(tuned_bytes) / static_cast<float>(freq_bytes));
    set_access_window(
        stream,
        data_region,
        tuned_bytes,
        tuned_hit_ratio,
        cudaAccessPropertyPersisting,
        cudaAccessPropertyStreaming);
    const float tuned_ms = measure_case_ms(stream, data_region, sink, total_elems, freq_elems);

    CHECK_CUDA(cudaMemcpy(
        host_sink.data(),
        sink,
        static_cast<size_t>(freq_elems) * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));
    const bool sink_ok = validate_sink(host_sink);
    all_ok = all_ok && sink_ok;

    std::cout << freq_mb << ' ' << streaming_mb << ' ' << baseline_ms << ' ' << fixed_ms << ' '
              << tuned_ms << ' ' << (baseline_ms / fixed_ms) << ' '
              << (baseline_ms / tuned_ms) << ' ' << (sink_ok ? "PASS" : "FAIL") << '\n';

    clear_access_window(stream);
  }

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(sink));
  CHECK_CUDA(cudaFree(data_region));
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
