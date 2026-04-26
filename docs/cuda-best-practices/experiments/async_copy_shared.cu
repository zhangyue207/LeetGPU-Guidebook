#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

constexpr int k_block_size = 256;
constexpr int k_items_per_thread = 8;
constexpr int k_repeats = 200;
constexpr int k_blocks = 4096;
constexpr int k_warmup = 10;
constexpr float k_relative_tolerance = 1e-4f;
constexpr float k_pair_tolerance = 1e-6f;

struct alignas(8) int2x {
  int x;
  int y;
};

struct alignas(16) int4x {
  int x;
  int y;
  int z;
  int w;
};

template <typename T>
__host__ __device__ __forceinline__ T add_value(const T& value, int delta);

template <>
__host__ __device__ __forceinline__ int add_value<int>(const int& value, int delta) {
  return value + delta;
}

template <>
__host__ __device__ __forceinline__ int2x add_value<int2x>(const int2x& value, int delta) {
  return int2x{value.x + delta, value.y + delta};
}

template <>
__host__ __device__ __forceinline__ int4x add_value<int4x>(const int4x& value, int delta) {
  return int4x{value.x + delta, value.y + delta, value.z + delta, value.w + delta};
}

template <typename T>
__host__ __device__ __forceinline__ float reduce_value(const T& value);

template <>
__host__ __device__ __forceinline__ float reduce_value<int>(const int& value) {
  return static_cast<float>(value);
}

template <>
__host__ __device__ __forceinline__ float reduce_value<int2x>(const int2x& value) {
  return static_cast<float>(value.x + value.y);
}

template <>
__host__ __device__ __forceinline__ float reduce_value<int4x>(const int4x& value) {
  return static_cast<float>(value.x + value.y + value.z + value.w);
}

template <typename T>
__global__ void init_input_kernel(T* data, int total_items) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_items) {
    return;
  }

  T value{};
  const int base = idx * 7 + 3;
  if constexpr (sizeof(T) == sizeof(int)) {
    value = base;
  } else if constexpr (sizeof(T) == sizeof(int2x)) {
    value = int2x{base, base + 1};
  } else {
    value = int4x{base, base + 1, base + 2, base + 3};
  }
  data[idx] = value;
}

template <typename T>
__global__ void sync_copy_kernel(const T* input, float* output, int items_per_block, int iterations) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  T* shared = reinterpret_cast<T*>(shared_bytes);

  const int linear_tid = threadIdx.x;
  const int block_offset = blockIdx.x * items_per_block;
  const int thread_offset = linear_tid * k_items_per_thread;

  float accum = 0.0f;
  for (int iter = 0; iter < iterations; ++iter) {
    #pragma unroll
    for (int i = 0; i < k_items_per_thread; ++i) {
      const int index = block_offset + thread_offset + i;
      T value = add_value(input[index], iter);
      shared[thread_offset + i] = value;
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < k_items_per_thread; ++i) {
      accum += reduce_value(shared[thread_offset + ((i + iter) & (k_items_per_thread - 1))]);
    }

    __syncthreads();
  }

  output[blockIdx.x * blockDim.x + linear_tid] = accum;
}

template <typename T>
__global__ void async_copy_kernel(const T* input, float* output, int items_per_block, int iterations) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  T* shared = reinterpret_cast<T*>(shared_bytes);

  const int linear_tid = threadIdx.x;
  const int block_offset = blockIdx.x * items_per_block;
  const int thread_offset = linear_tid * k_items_per_thread;

  float accum = 0.0f;
  for (int iter = 0; iter < iterations; ++iter) {
    #pragma unroll
    for (int i = 0; i < k_items_per_thread; ++i) {
      const int index = block_offset + thread_offset + i;
#if __CUDA_ARCH__ >= 800
      __pipeline_memcpy_async(
          &shared[thread_offset + i],
          &input[index],
          sizeof(T));
#else
      shared[thread_offset + i] = input[index];
#endif
    }
#if __CUDA_ARCH__ >= 800
    __pipeline_commit();
    __pipeline_wait_prior(0);
#endif
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < k_items_per_thread; ++i) {
      accum += reduce_value(add_value(shared[thread_offset + ((i + iter) & (k_items_per_thread - 1))], iter));
    }

    __syncthreads();
  }

  output[blockIdx.x * blockDim.x + linear_tid] = accum;
}

template <typename T>
double cpu_reference_for_thread(const std::vector<T>& input, int block_idx, int thread_idx) {
  const int items_per_block = k_block_size * k_items_per_thread;
  const int block_offset = block_idx * items_per_block;
  const int thread_offset = thread_idx * k_items_per_thread;

  double accum = 0.0;
  for (int iter = 0; iter < k_repeats; ++iter) {
    for (int i = 0; i < k_items_per_thread; ++i) {
      const int index = block_offset + thread_offset + ((i + iter) & (k_items_per_thread - 1));
      accum += static_cast<double>(reduce_value(add_value(input[index], iter)));
    }
  }
  return accum;
}

template <typename T>
bool validate_reference_output(const std::vector<T>& input, const std::vector<float>& output) {
  const int sample_blocks[] = {0, k_blocks / 2, k_blocks - 1};
  const int sample_threads[] = {0, k_block_size / 2, k_block_size - 1};
  for (int block_idx : sample_blocks) {
    for (int thread_idx : sample_threads) {
      const int out_idx = block_idx * k_block_size + thread_idx;
      const double expected = cpu_reference_for_thread(input, block_idx, thread_idx);
      const double actual = output[out_idx];
      if (std::fabs(actual - expected) > k_relative_tolerance * std::max(1.0, std::fabs(expected))) {
        std::cerr << "Validation failed at block " << block_idx
                  << ", thread " << thread_idx
                  << ": expected " << expected
                  << ", got " << actual << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool validate_pair_output(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  const int sample_indices[] = {
      0,
      static_cast<int>(lhs.size() / 3),
      static_cast<int>((lhs.size() * 2) / 3),
      static_cast<int>(lhs.size() - 1)};
  for (int idx : sample_indices) {
    if (std::fabs(lhs[idx] - rhs[idx]) > k_pair_tolerance * std::max(1.0f, std::fabs(lhs[idx]))) {
      std::cerr << "Sync/async mismatch at index " << idx
                << ": sync=" << lhs[idx]
                << ", async=" << rhs[idx] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename LaunchFn, typename T>
float measure_kernel_ms(LaunchFn&& launch, const T* input, float* output) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < k_warmup; ++i) {
    launch(input, output, k_block_size * k_items_per_thread, 1);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  launch(input, output, k_block_size * k_items_per_thread, k_repeats);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return elapsed_ms;
}

template <typename T>
void launch_sync(const T* input, float* output, int items_per_block, int iterations, size_t shared_bytes) {
  sync_copy_kernel<T><<<k_blocks, k_block_size, shared_bytes>>>(input, output, items_per_block, iterations);
}

template <typename T>
void launch_async(const T* input, float* output, int items_per_block, int iterations, size_t shared_bytes) {
  async_copy_kernel<T><<<k_blocks, k_block_size, shared_bytes>>>(input, output, items_per_block, iterations);
}

struct ResultRow {
  std::string type_name;
  int bytes_per_copy;
  float sync_ms;
  float async_ms;
  float speedup;
  bool sync_ok;
  bool async_ok;
};

template <typename T>
ResultRow run_experiment(const std::string& type_name) {
  const int items_per_block = k_block_size * k_items_per_thread;
  const int total_items = k_blocks * items_per_block;
  const size_t input_bytes = static_cast<size_t>(total_items) * sizeof(T);
  const size_t output_items = static_cast<size_t>(k_blocks) * k_block_size;
  const size_t output_bytes = output_items * sizeof(float);
  const size_t shared_bytes = static_cast<size_t>(items_per_block) * sizeof(T);

  T* d_input = nullptr;
  float* d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
  CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

  init_input_kernel<<<(total_items + 255) / 256, 256>>>(d_input, total_items);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  const float sync_ms = measure_kernel_ms(
      [&](const T* input, float* output, int block_items, int iterations) {
        launch_sync<T>(input, output, block_items, iterations, shared_bytes);
      },
      d_input,
      d_output);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> h_sync(output_items);
  CHECK_CUDA(cudaMemcpy(h_sync.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

  const float async_ms = measure_kernel_ms(
      [&](const T* input, float* output, int block_items, int iterations) {
        launch_async<T>(input, output, block_items, iterations, shared_bytes);
      },
      d_input,
      d_output);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> h_async(output_items);
  CHECK_CUDA(cudaMemcpy(h_async.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
  std::vector<T> h_input(total_items);
  CHECK_CUDA(cudaMemcpy(h_input.data(), d_input, input_bytes, cudaMemcpyDeviceToHost));

  const bool sync_ref_ok = validate_reference_output(h_input, h_sync);
  const bool async_ref_ok = validate_reference_output(h_input, h_async);
  const bool pair_ok = validate_pair_output(h_sync, h_async);
  const bool sync_ok = sync_ref_ok && pair_ok;
  const bool async_ok = async_ref_ok && pair_ok;

  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_input));

  return ResultRow{
      type_name,
      static_cast<int>(sizeof(T)),
      sync_ms,
      async_ms,
      sync_ms / async_ms,
      sync_ok,
      async_ok};
}

void print_device_info() {
  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "gpu: " << prop.name << '\n';
  std::cout << "cuda_capability: " << prop.major << "." << prop.minor << '\n';
}

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(3);
  print_device_info();
  std::cout << "blocks: " << k_blocks << '\n';
  std::cout << "block_size: " << k_block_size << '\n';
  std::cout << "items_per_thread: " << k_items_per_thread << '\n';
  std::cout << "repeats: " << k_repeats << '\n';

  std::vector<ResultRow> results;
  results.push_back(run_experiment<int>("int"));
  results.push_back(run_experiment<int2x>("int2"));
  results.push_back(run_experiment<int4x>("int4"));

  bool all_ok = true;
  std::cout << "type bytes_per_copy sync_ms async_ms speedup sync_check async_check" << '\n';
  for (const auto& row : results) {
    std::cout << row.type_name << ' '
              << row.bytes_per_copy << ' '
              << row.sync_ms << ' '
              << row.async_ms << ' '
              << row.speedup << ' '
              << (row.sync_ok ? "PASS" : "FAIL") << ' '
              << (row.async_ok ? "PASS" : "FAIL") << '\n';
    all_ok = all_ok && row.sync_ok && row.async_ok;
  }

  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
