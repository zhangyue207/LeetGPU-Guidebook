#include <cuda_runtime.h>

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

constexpr int kBlockSize = 256;
constexpr int kDivergenceElems = 1 << 24;
constexpr int kPredicationElems = 1 << 24;
constexpr int kSchedulerElems = 1 << 24;
constexpr int kSchedulerExtraSmem[] = {0, 8 << 10, 32 << 10};
constexpr int kPredicationRepeats = 512;

enum DivergencePattern {
  kUniformTrue = 0,
  kWarpAlignedHalf = 1,
  kCheckerboard = 2,
  kRandom50 = 3,
  kRandom10 = 4,
};

enum PredicationPattern {
  kPredUniform = 0,
  kPredCheckerboard = 1,
  kPredRandom50 = 2,
};

__device__ __forceinline__ float heavy_path(float x, int ops) {
  #pragma unroll 1
  for (int i = 0; i < ops; ++i) {
    x = fmaf(x, 1.0001f, 0.1234f);
  }
  return x;
}

__device__ __forceinline__ bool divergence_predicate(int pattern, int idx) {
  const int lane = idx & 31;
  switch (pattern) {
    case kUniformTrue:
      return true;
    case kWarpAlignedHalf:
      return lane < 16;
    case kCheckerboard:
      return (lane & 1) == 0;
    case kRandom50:
      return ((idx * 17 + 23) & 31) < 16;
    case kRandom10:
      return ((idx * 13 + 7) % 10) == 0;
    default:
      return true;
  }
}

__device__ __forceinline__ bool predication_predicate(int pattern, int idx) {
  switch (pattern) {
    case kPredUniform:
      return true;
    case kPredCheckerboard:
      return (idx & 1) == 0;
    case kPredRandom50:
      return ((idx * 17 + 23) & 31) < 16;
    default:
      return true;
  }
}

__global__ void divergence_kernel(float* out, int n, int pattern, int ops) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float x = static_cast<float>((idx & 255) + 1) * 0.001f;
  if (divergence_predicate(pattern, idx)) {
    x = heavy_path(x, ops);
  } else {
    x = heavy_path(x + 0.5f, ops);
  }
  out[idx] = x;
}

__device__ __forceinline__ float short_add_path(float x, int ops) {
  #pragma unroll 1
  for (int i = 0; i < ops; ++i) {
    x = fmaf(x, 1.0001f, 0.001f * static_cast<float>(i + 1));
  }
  return x;
}

__device__ __forceinline__ float short_sub_path(float x, int ops) {
  #pragma unroll 1
  for (int i = 0; i < ops; ++i) {
    x = fmaf(x, 0.9999f, -0.001f * static_cast<float>(i + 1));
  }
  return x;
}

__global__ void branch_kernel(float* out, int n, int pattern, int ops, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float x = static_cast<float>((idx & 255) + 1) * 0.001f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    if (predication_predicate(pattern, idx)) {
      x = short_add_path(x, ops);
    }
  }
  out[idx] = x;
}

__global__ void compute_both_kernel(float* out, int n, int pattern, int ops, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float x = static_cast<float>((idx & 255) + 1) * 0.001f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    const bool pred = predication_predicate(pattern, idx);
    const float on_true = short_add_path(x, ops);
    const float on_false = short_sub_path(x, ops);
    x = pred ? on_true : on_false;
  }
  out[idx] = x;
}

__global__ void scheduler_dep_kernel(float* out, int n, int iters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float smem[];
  if (idx >= n) {
    return;
  }
  if (threadIdx.x < blockDim.x) {
    smem[threadIdx.x] = static_cast<float>(threadIdx.x);
  }
  __syncthreads();
  float x = static_cast<float>((idx & 255) + 1) * 0.001f;
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const float bias = smem[threadIdx.x] * 1.0e-6f;
    x = fmaf(x, 1.0001f, 0.1234f + bias);
    x = fmaf(x, 1.0002f, 0.1134f + bias);
    x = fmaf(x, 0.9999f, 0.1034f + bias);
    x = fmaf(x, 1.0003f, 0.0934f + bias);
  }
  out[idx] = x;
}

__global__ void scheduler_ilp4_kernel(float* out, int n, int iters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float smem[];
  if (idx >= n) {
    return;
  }
  if (threadIdx.x < blockDim.x) {
    smem[threadIdx.x] = static_cast<float>(threadIdx.x);
  }
  __syncthreads();
  float x0 = static_cast<float>((idx & 255) + 1) * 0.001f;
  float x1 = x0 + 0.1f;
  float x2 = x0 + 0.2f;
  float x3 = x0 + 0.3f;
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const float bias = smem[threadIdx.x] * 1.0e-6f;
    x0 = fmaf(x0, 1.0001f, 0.1234f + bias);
    x1 = fmaf(x1, 1.0002f, 0.1134f + bias);
    x2 = fmaf(x2, 0.9999f, 0.1034f + bias);
    x3 = fmaf(x3, 1.0003f, 0.0934f + bias);
  }
  out[idx] = x0 + x1 + x2 + x3;
}

float measure_ms(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

template <typename LaunchFn>
float run_timed(LaunchFn launch, cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  launch();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(start, stream));
  launch();
  CHECK_CUDA(cudaEventRecord(stop, stream));
  return measure_ms(stream, start, stop);
}

void run_divergence(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(kDivergenceElems) * sizeof(float)));
  const int grid = (kDivergenceElems + kBlockSize - 1) / kBlockSize;
  const int ops = 64;

  std::cout << "experiment: warp_divergence\n";
  std::cout << "pattern kernel_ms speedup_vs_uniform\n";

  const char* names[] = {"uniform_true", "warp_aligned_half", "checkerboard", "random_50", "random_10"};
  float baseline = 0.0f;
  for (int pattern = 0; pattern <= kRandom10; ++pattern) {
    const float ms = run_timed(
        [&] {
          divergence_kernel<<<grid, kBlockSize, 0, stream>>>(d_out, kDivergenceElems, pattern, ops);
        },
        stream,
        start,
        stop);
    if (pattern == 0) {
      baseline = ms;
    }
    std::cout << std::fixed << std::setprecision(6)
              << names[pattern] << ' ' << ms << ' ' << (baseline / ms) << '\n';
  }

  CHECK_CUDA(cudaFree(d_out));
}

void run_predication(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(kPredicationElems) * sizeof(float)));
  const int grid = (kPredicationElems + kBlockSize - 1) / kBlockSize;
  const char* pattern_names[] = {"uniform", "checkerboard", "random_50"};
  const int ops_list[] = {1, 2, 4, 8, 16, 32};

  std::cout << "experiment: branch_vs_compute_both\n";
  std::cout << "pattern body_ops branch_ms compute_both_ms winner\n";

  for (int pattern = 0; pattern <= kPredRandom50; ++pattern) {
    for (int ops : ops_list) {
      const float branch_ms = run_timed(
          [&] {
            branch_kernel<<<grid, kBlockSize, 0, stream>>>(d_out, kPredicationElems, pattern, ops, kPredicationRepeats);
          },
          stream,
          start,
          stop);
      const float compute_both_ms = run_timed(
          [&] {
            compute_both_kernel<<<grid, kBlockSize, 0, stream>>>(d_out, kPredicationElems, pattern, ops, kPredicationRepeats);
          },
          stream,
          start,
          stop);
      std::cout << std::fixed << std::setprecision(6)
                << pattern_names[pattern] << ' ' << ops << ' '
                << branch_ms << ' ' << compute_both_ms << ' '
                << (branch_ms <= compute_both_ms ? "branch" : "compute_both") << '\n';
    }
  }

  CHECK_CUDA(cudaFree(d_out));
}

void run_scheduler(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(kSchedulerElems) * sizeof(float)));
  const int block_size = 256;
  const int iters = 256;

  std::cout << "experiment: scheduler_latency_hiding\n";
  std::cout << "kernel extra_smem_bytes theoretical_occupancy kernel_ms speedup_vs_dep_0KB\n";

  const int grid = (kSchedulerElems + block_size - 1) / block_size;
  float dep0 = 0.0f;
  for (int extra_smem : kSchedulerExtraSmem) {
    const size_t total_smem = static_cast<size_t>(block_size * sizeof(float) + extra_smem);
    int active_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        scheduler_dep_kernel,
        block_size,
        total_smem));
    const float occupancy =
        static_cast<float>(active_blocks * block_size) / 2048.0f;

    const float dep_ms = run_timed(
        [&] {
          scheduler_dep_kernel<<<grid, block_size, total_smem, stream>>>(d_out, kSchedulerElems, iters);
        },
        stream,
        start,
        stop);
    if (extra_smem == 0) {
      dep0 = dep_ms;
    }
    const float ilp_ms = run_timed(
        [&] {
          scheduler_ilp4_kernel<<<grid, block_size, total_smem, stream>>>(d_out, kSchedulerElems, iters);
        },
        stream,
        start,
        stop);
    std::cout << std::fixed << std::setprecision(6)
              << "dep_chain " << extra_smem << ' ' << occupancy << ' ' << dep_ms << ' ' << (dep0 / dep_ms) << '\n';
    std::cout << std::fixed << std::setprecision(6)
              << "ilp4 " << extra_smem << ' ' << occupancy << ' ' << ilp_ms << ' ' << (dep0 / ilp_ms) << '\n';
  }

  CHECK_CUDA(cudaFree(d_out));
}

}  // namespace

int main(int argc, char** argv) {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  if (argc > 1 && std::string(argv[1]) == "profile") {
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(kSchedulerElems) * sizeof(float)));
    const int block_size = 256;
    const int grid = (kSchedulerElems + block_size - 1) / block_size;
    const int ops = 64;
    const int pred_ops = 8;
    const int iters = 256;

    const std::string target = argc > 2 ? argv[2] : "divergence";
    if (target == "divergence") {
      const int pattern = argc > 3 ? std::atoi(argv[3]) : kCheckerboard;
      divergence_kernel<<<grid, block_size, 0, stream>>>(d_out, kDivergenceElems, pattern, ops);
    } else if (target == "branch") {
      const int pattern = argc > 3 ? std::atoi(argv[3]) : kPredRandom50;
      const int body_ops = argc > 4 ? std::atoi(argv[4]) : pred_ops;
      branch_kernel<<<grid, block_size, 0, stream>>>(d_out, kPredicationElems, pattern, body_ops, kPredicationRepeats);
    } else if (target == "compute_both") {
      const int pattern = argc > 3 ? std::atoi(argv[3]) : kPredRandom50;
      const int body_ops = argc > 4 ? std::atoi(argv[4]) : pred_ops;
      compute_both_kernel<<<grid, block_size, 0, stream>>>(d_out, kPredicationElems, pattern, body_ops, kPredicationRepeats);
    } else if (target == "dep") {
      const int extra_smem = argc > 3 ? std::atoi(argv[3]) : (32 << 10);
      const size_t total_smem = static_cast<size_t>(block_size * sizeof(float) + extra_smem);
      scheduler_dep_kernel<<<grid, block_size, total_smem, stream>>>(d_out, kSchedulerElems, iters);
    } else if (target == "ilp4") {
      const int extra_smem = argc > 3 ? std::atoi(argv[3]) : (32 << 10);
      const size_t total_smem = static_cast<size_t>(block_size * sizeof(float) + extra_smem);
      scheduler_ilp4_kernel<<<grid, block_size, total_smem, stream>>>(d_out, kSchedulerElems, iters);
    } else {
      std::cerr << "unknown profile target: " << target << std::endl;
      return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
  }

  run_divergence(stream, start, stop);
  run_predication(stream, start, stop);
  run_scheduler(stream, start, stop);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
