#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
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
constexpr int k_num_streams = 4;
constexpr int k_repeats = 20;
constexpr int k_compute_iters = 96;
constexpr size_t k_total_bytes = 128u << 20;

__global__ void compute_kernel(float* data, int n, int compute_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = data[idx];
    for (int i = 0; i < compute_iters; ++i) {
      x = fmaf(x, 1.000001f, 1.0f);
    }
    data[idx] = x;
  }
}

double run_single_stream(
    const float* host_in,
    float* host_out,
    float* device_buffer,
    int total_elems,
    int chunk_elems,
    cudaStream_t stream) {
  CHECK_CUDA(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int offset = 0; offset < total_elems; offset += chunk_elems) {
    CHECK_CUDA(cudaMemcpyAsync(
        device_buffer + offset,
        host_in + offset,
        static_cast<size_t>(chunk_elems) * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
    compute_kernel<<<(chunk_elems + k_block_size - 1) / k_block_size, k_block_size, 0, stream>>>(
        device_buffer + offset, chunk_elems, k_compute_iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(
        host_out + offset,
        device_buffer + offset,
        static_cast<size_t>(chunk_elems) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_multi_stream(
    const float* host_in,
    float* host_out,
    float* device_buffer,
    int total_elems,
    int chunk_elems,
    cudaStream_t* streams,
    int num_streams) {
  CHECK_CUDA(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < num_streams; ++i) {
    int offset = i * chunk_elems;
    CHECK_CUDA(cudaMemcpyAsync(
        device_buffer + offset,
        host_in + offset,
        static_cast<size_t>(chunk_elems) * sizeof(float),
        cudaMemcpyHostToDevice,
        streams[i]));
    compute_kernel<<<(chunk_elems + k_block_size - 1) / k_block_size, k_block_size, 0, streams[i]>>>(
        device_buffer + offset, chunk_elems, k_compute_iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(
        host_out + offset,
        device_buffer + offset,
        static_cast<size_t>(chunk_elems) * sizeof(float),
        cudaMemcpyDeviceToHost,
        streams[i]));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

float reference_value(float x) {
  for (int i = 0; i < k_compute_iters; ++i) {
    x = std::fma(x, 1.000001f, 1.0f);
  }
  return x;
}

bool validate_samples(const float* host_in, const float* host_out, int total_elems) {
  const int sample_indices[] = {0, total_elems / 3, (2 * total_elems) / 3, total_elems - 1};
  for (int idx : sample_indices) {
    float expected = reference_value(host_in[idx]);
    float actual = host_out[idx];
    if (std::fabs(expected - actual) > 1e-3f) {
      std::cerr << "Validation failed at " << idx << ": expected " << expected
                << ", got " << actual << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const int total_elems = static_cast<int>(k_total_bytes / sizeof(float));
  const int chunk_elems = total_elems / k_num_streams;

  float* host_in = nullptr;
  float* host_out = nullptr;
  float* device_buffer = nullptr;
  CHECK_CUDA(cudaMallocHost(&host_in, k_total_bytes));
  CHECK_CUDA(cudaMallocHost(&host_out, k_total_bytes));
  CHECK_CUDA(cudaMalloc(&device_buffer, k_total_bytes));

  for (int i = 0; i < total_elems; ++i) {
    host_in[i] = static_cast<float>(i % 1024) * 0.5f;
    host_out[i] = 0.0f;
  }

  cudaStream_t single_stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&single_stream, cudaStreamNonBlocking));
  cudaStream_t streams[k_num_streams];
  for (int i = 0; i < k_num_streams; ++i) {
    CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }

  run_single_stream(host_in, host_out, device_buffer, total_elems, chunk_elems, single_stream);
  run_multi_stream(host_in, host_out, device_buffer, total_elems, chunk_elems, streams, k_num_streams);

  double single_total_ms = 0.0;
  double multi_total_ms = 0.0;
  for (int repeat = 0; repeat < k_repeats; ++repeat) {
    single_total_ms +=
        run_single_stream(host_in, host_out, device_buffer, total_elems, chunk_elems, single_stream);
    multi_total_ms += run_multi_stream(
        host_in, host_out, device_buffer, total_elems, chunk_elems, streams, k_num_streams);
  }

  const bool valid = validate_samples(host_in, host_out, total_elems);
  const double single_avg_ms = single_total_ms / k_repeats;
  const double multi_avg_ms = multi_total_ms / k_repeats;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "total_mb: " << (k_total_bytes >> 20) << '\n';
  std::cout << "streams: " << k_num_streams << '\n';
  std::cout << "compute_iters: " << k_compute_iters << '\n';
  std::cout << "repeats: " << k_repeats << '\n';
  std::cout << "single_stream_ms_avg: " << single_avg_ms << '\n';
  std::cout << "multi_stream_ms_avg: " << multi_avg_ms << '\n';
  std::cout << "speedup: " << (single_avg_ms / multi_avg_ms) << '\n';
  std::cout << "validation: " << (valid ? "PASS" : "FAIL") << '\n';

  for (int i = 0; i < k_num_streams; ++i) {
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
  CHECK_CUDA(cudaStreamDestroy(single_stream));
  CHECK_CUDA(cudaFree(device_buffer));
  CHECK_CUDA(cudaFreeHost(host_in));
  CHECK_CUDA(cudaFreeHost(host_out));
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
