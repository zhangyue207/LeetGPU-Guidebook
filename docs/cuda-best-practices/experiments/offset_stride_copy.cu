#include <cuda_runtime.h>

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

constexpr int k_block_size = 256;
constexpr int k_repeats = 20;
constexpr size_t k_total_bytes = 64u << 20;
constexpr int k_offset_floats = 1;
constexpr int k_stride = 8;

__global__ void offset_copy_kernel(int* out, const int* in, int n, int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx + offset];
  }
}

__global__ void stride_copy_kernel(int* out, const int* in, int n, int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx * stride];
  }
}

float measure_kernel_ms(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  CHECK_CUDA(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  return elapsed_ms;
}

bool validate_contiguous(const int* in, const int* out, int n) {
  for (int i = 0; i < 4; ++i) {
    int idx = (n - 1) * i / 3;
    if (out[idx] != in[idx]) {
      return false;
    }
  }
  return true;
}

bool validate_offset(const int* in, const int* out, int n, int offset) {
  for (int i = 0; i < 4; ++i) {
    int idx = (n - 1) * i / 3;
    if (out[idx] != in[idx + offset]) {
      return false;
    }
  }
  return true;
}

bool validate_stride(const int* in, const int* out, int n, int stride) {
  for (int i = 0; i < 4; ++i) {
    int idx = (n - 1) * i / 3;
    if (out[idx] != in[idx * stride]) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int contiguous_elems = static_cast<int>(k_total_bytes / sizeof(float));
  constexpr int offset_elems = contiguous_elems;
  constexpr int stride_elems = contiguous_elems / k_stride;
  constexpr int source_elems = stride_elems * k_stride;

  int* in = nullptr;
  int* out = nullptr;
  CHECK_CUDA(cudaMalloc(&in, static_cast<size_t>(source_elems + k_offset_floats) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&out, static_cast<size_t>(contiguous_elems) * sizeof(int)));

  int* host = static_cast<int*>(std::malloc(static_cast<size_t>(source_elems + k_offset_floats) * sizeof(int)));
  if (host == nullptr) {
    std::cerr << "malloc failed" << std::endl;
    return EXIT_FAILURE;
  }
  for (int i = 0; i < source_elems + k_offset_floats; ++i) {
    host[i] = i;
  }
  CHECK_CUDA(cudaMemcpy(in, host, static_cast<size_t>(source_elems + k_offset_floats) * sizeof(int), cudaMemcpyHostToDevice));

  int* host_out = static_cast<int*>(std::malloc(static_cast<size_t>(contiguous_elems) * sizeof(int)));
  if (host_out == nullptr) {
    std::cerr << "malloc failed" << std::endl;
    return EXIT_FAILURE;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CHECK_CUDA(cudaMemcpy(out, in, static_cast<size_t>(contiguous_elems) * sizeof(int), cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < k_repeats; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(
        out,
        in,
        static_cast<size_t>(contiguous_elems) * sizeof(int),
        cudaMemcpyDeviceToDevice,
        stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const float contiguous_ms = measure_kernel_ms(stream, start, stop) / k_repeats;

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < k_repeats; ++i) {
    offset_copy_kernel<<<(offset_elems + k_block_size - 1) / k_block_size, k_block_size, 0, stream>>>(
        out, in, offset_elems, k_offset_floats);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const float offset_ms = measure_kernel_ms(stream, start, stop) / k_repeats;

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < k_repeats; ++i) {
    stride_copy_kernel<<<(stride_elems + k_block_size - 1) / k_block_size, k_block_size, 0, stream>>>(
        out, in, stride_elems, k_stride);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const float stride_ms = measure_kernel_ms(stream, start, stop) / k_repeats;

  CHECK_CUDA(cudaMemcpyAsync(
      out,
      in,
      static_cast<size_t>(contiguous_elems) * sizeof(int),
      cudaMemcpyDeviceToDevice,
      stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(
      host_out,
      out,
      static_cast<size_t>(contiguous_elems) * sizeof(int),
      cudaMemcpyDeviceToHost));
  const bool contiguous_ok = validate_contiguous(host, host_out, contiguous_elems);

  offset_copy_kernel<<<(offset_elems + k_block_size - 1) / k_block_size, k_block_size, 0, stream>>>(
      out, in, offset_elems, k_offset_floats);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(
      host_out,
      out,
      static_cast<size_t>(offset_elems) * sizeof(int),
      cudaMemcpyDeviceToHost));
  const bool offset_ok = validate_offset(host, host_out, offset_elems, k_offset_floats);

  stride_copy_kernel<<<(stride_elems + k_block_size - 1) / k_block_size, k_block_size, 0, stream>>>(
      out, in, stride_elems, k_stride);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(
      host_out,
      out,
      static_cast<size_t>(stride_elems) * sizeof(int),
      cudaMemcpyDeviceToHost));
  const bool stride_ok = validate_stride(host, host_out, stride_elems, k_stride);

  const double contiguous_gbps =
      static_cast<double>(contiguous_elems) * sizeof(float) / (contiguous_ms * 1.0e-3) / 1.0e9;
  const double offset_gbps =
      static_cast<double>(offset_elems) * sizeof(float) / (offset_ms * 1.0e-3) / 1.0e9;
  const double stride_gbps =
      static_cast<double>(stride_elems) * sizeof(float) / (stride_ms * 1.0e-3) / 1.0e9;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "total_mb: " << (k_total_bytes >> 20) << '\n';
  std::cout << "repeats: " << k_repeats << '\n';
  std::cout << "offset_floats: " << k_offset_floats << '\n';
  std::cout << "stride: " << k_stride << '\n';
  std::cout << "pattern contiguous_GBps offset_GBps stride_GBps contiguous_check offset_check stride_check"
            << '\n';
  std::cout << "bandwidth " << contiguous_gbps << ' ' << offset_gbps << ' ' << stride_gbps
            << ' ' << (contiguous_ok ? "PASS" : "FAIL")
            << ' ' << (offset_ok ? "PASS" : "FAIL")
            << ' ' << (stride_ok ? "PASS" : "FAIL") << '\n';

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(in));
  CHECK_CUDA(cudaFree(out));
  std::free(host);
  std::free(host_out);
  return (contiguous_ok && offset_ok && stride_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
