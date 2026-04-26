#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
constexpr int kRepeats = 100;

constexpr int kHalfElems = 1 << 24;
constexpr int kDivElems = 1 << 24;
constexpr int kFdivElems = 1 << 24;
constexpr int kNormElems = 1 << 24;
constexpr int kRsqrtElems = 1 << 24;
constexpr int kDivShift = 5;
constexpr int kDivPow2 = 1 << kDivShift;

struct TimingResult {
  float ms = 0.0f;
};

struct ErrorStats {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double mean_rel_error = 0.0;
};

__global__ void half_axpy_float_kernel(float* out, const float* x, float a, float b, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  float v = x[idx];
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    v = fmaf(v, a, b);
  }
  out[idx] = v;
}

__global__ void half_axpy_half_kernel(__half* out, const __half* x, __half a, __half b, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  __half v = x[idx];
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    v = __hadd(__hmul(v, a), b);
  }
  out[idx] = v;
}

__global__ void half_axpy_half2_kernel(__half2* out, const __half2* x, __half2 a, __half2 b, int n2, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n2) {
    return;
  }
  __half2 v = x[idx];
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    v = __hadd2(__hmul2(v, a), b);
  }
  out[idx] = v;
}

__global__ void div_kernel(int* q, int n, int divisor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  q[idx] = idx / divisor;
}

__global__ void shift_kernel(int* q, int n, int shift) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  q[idx] = idx >> shift;
}

__global__ void mod_kernel(int* r, int n, int divisor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  r[idx] = idx % divisor;
}

__global__ void mask_kernel(int* r, int n, int mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  r[idx] = idx & mask;
}

__global__ void fdiv_precise_kernel(float* out, const float* x, const float* y, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float xv = x[idx];
  const float yv = y[idx];
  float acc = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    acc += xv / (yv + 1.0e-7f * static_cast<float>(i));
  }
  out[idx] = acc;
}

__global__ void fdiv_fast_kernel(float* out, const float* x, const float* y, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float xv = x[idx];
  const float yv = y[idx];
  float acc = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    acc += __fdividef(xv, yv + 1.0e-7f * static_cast<float>(i));
  }
  out[idx] = acc;
}

__global__ void fdiv_rcp_kernel(float* out, const float* x, const float* y, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float xv = x[idx];
  const float yv = y[idx];
  float acc = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    const float denom = yv + 1.0e-7f * static_cast<float>(i);
    acc += xv * __frcp_rn(denom);
  }
  out[idx] = acc;
}

__global__ void rsqrt_precise_kernel(float* out, const float* x, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float base = x[idx];
  float v = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    v += 1.0f / sqrtf(base + 1.0e-6f + 1.0e-7f * static_cast<float>(i));
  }
  out[idx] = v;
}

__global__ void rsqrt_fast_kernel(float* out, const float* x, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float base = x[idx];
  float v = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    v += rsqrtf(base + 1.0e-6f + 1.0e-7f * static_cast<float>(i));
  }
  out[idx] = v;
}

__global__ void normalize3_precise_kernel(float* out, const float* x, const float* y, const float* z, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float xv = x[idx];
  const float yv = y[idx];
  const float zv = z[idx];
  float acc = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    const float len = sqrtf(xv * xv + yv * yv + zv * zv + 1.0e-12f);
    const float inv = 1.0f / len;
    acc += xv * inv + yv * inv + zv * inv;
  }
  out[idx] = acc;
}

__global__ void normalize3_rsqrt_kernel(float* out, const float* x, const float* y, const float* z, int n, int repeats) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float xv = x[idx];
  const float yv = y[idx];
  const float zv = z[idx];
  float acc = 0.0f;
  #pragma unroll 1
  for (int i = 0; i < repeats; ++i) {
    const float inv = rsqrtf(xv * xv + yv * yv + zv * zv + 1.0e-12f);
    acc += xv * inv + yv * inv + zv * inv;
  }
  out[idx] = acc;
}

TimingResult measure_events(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  CHECK_CUDA(cudaEventSynchronize(stop));
  TimingResult result;
  CHECK_CUDA(cudaEventElapsedTime(&result.ms, start, stop));
  return result;
}

ErrorStats compare_float_vectors(const std::vector<float>& ref, const std::vector<float>& actual) {
  ErrorStats stats;
  double sum_rel = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double abs_err = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(ref[i]));
    const double rel_err = abs_err / std::max(1.0, std::fabs(static_cast<double>(ref[i])));
    stats.max_abs_error = std::max(stats.max_abs_error, abs_err);
    stats.max_rel_error = std::max(stats.max_rel_error, rel_err);
    sum_rel += rel_err;
  }
  stats.mean_rel_error = sum_rel / static_cast<double>(ref.size());
  return stats;
}

bool validate_int_vectors(const std::vector<int>& lhs, const std::vector<int>& rhs) {
  const int sample_indices[] = {0, static_cast<int>(lhs.size() / 3), static_cast<int>((lhs.size() * 2) / 3), static_cast<int>(lhs.size() - 1)};
  for (int idx : sample_indices) {
    if (lhs[idx] != rhs[idx]) {
      return false;
    }
  }
  return true;
}

void run_half2_experiment(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  std::vector<float> host_float_in(static_cast<size_t>(kHalfElems));
  for (int i = 0; i < kHalfElems; ++i) {
    host_float_in[static_cast<size_t>(i)] = static_cast<float>((i % 1024) - 512) * 0.001f;
  }

  std::vector<__half> host_half_in(static_cast<size_t>(kHalfElems));
  for (int i = 0; i < kHalfElems; ++i) {
    host_half_in[static_cast<size_t>(i)] = __float2half(host_float_in[static_cast<size_t>(i)]);
  }

  float *d_float_in = nullptr, *d_float_out = nullptr;
  __half *d_half_in = nullptr, *d_half_out = nullptr;
  __half2 *d_half2_in = nullptr, *d_half2_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_float_in, static_cast<size_t>(kHalfElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_float_out, static_cast<size_t>(kHalfElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_half_in, static_cast<size_t>(kHalfElems) * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_half_out, static_cast<size_t>(kHalfElems) * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_half2_in, static_cast<size_t>(kHalfElems / 2) * sizeof(__half2)));
  CHECK_CUDA(cudaMalloc(&d_half2_out, static_cast<size_t>(kHalfElems / 2) * sizeof(__half2)));

  CHECK_CUDA(cudaMemcpy(d_float_in, host_float_in.data(), static_cast<size_t>(kHalfElems) * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_half_in, host_half_in.data(), static_cast<size_t>(kHalfElems) * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_half2_in, host_half_in.data(), static_cast<size_t>(kHalfElems) * sizeof(__half), cudaMemcpyHostToDevice));

  const int grid = (kHalfElems + kBlockSize - 1) / kBlockSize;
  const int grid_half2 = ((kHalfElems / 2) + kBlockSize - 1) / kBlockSize;
  const float a = 1.0005f;
  const float b = 0.0001f;

  half_axpy_float_kernel<<<grid, kBlockSize, 0, stream>>>(d_float_out, d_float_in, a, b, kHalfElems, kRepeats);
  half_axpy_half_kernel<<<grid, kBlockSize, 0, stream>>>(d_half_out, d_half_in, __float2half(a), __float2half(b), kHalfElems, kRepeats);
  half_axpy_half2_kernel<<<grid_half2, kBlockSize, 0, stream>>>(
      d_half2_out,
      d_half2_in,
      __halves2half2(__float2half(a), __float2half(a)),
      __halves2half2(__float2half(b), __float2half(b)),
      kHalfElems / 2,
      kRepeats);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start, stream));
  half_axpy_float_kernel<<<grid, kBlockSize, 0, stream>>>(d_float_out, d_float_in, a, b, kHalfElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult float_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  half_axpy_half_kernel<<<grid, kBlockSize, 0, stream>>>(d_half_out, d_half_in, __float2half(a), __float2half(b), kHalfElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult half_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  half_axpy_half2_kernel<<<grid_half2, kBlockSize, 0, stream>>>(
      d_half2_out,
      d_half2_in,
      __halves2half2(__float2half(a), __float2half(a)),
      __halves2half2(__float2half(b), __float2half(b)),
      kHalfElems / 2,
      kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult half2_time = measure_events(stream, start, stop);

  std::vector<float> host_float_out(static_cast<size_t>(kHalfElems));
  std::vector<__half> host_half_out(static_cast<size_t>(kHalfElems));
  std::vector<__half> host_half2_out(static_cast<size_t>(kHalfElems));
  CHECK_CUDA(cudaMemcpy(host_float_out.data(), d_float_out, static_cast<size_t>(kHalfElems) * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_half_out.data(), d_half_out, static_cast<size_t>(kHalfElems) * sizeof(__half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_half2_out.data(), d_half2_out, static_cast<size_t>(kHalfElems) * sizeof(__half), cudaMemcpyDeviceToHost));

  std::vector<float> half_as_float(static_cast<size_t>(kHalfElems));
  std::vector<float> half2_as_float(static_cast<size_t>(kHalfElems));
  for (int i = 0; i < kHalfElems; ++i) {
    half_as_float[static_cast<size_t>(i)] = __half2float(host_half_out[static_cast<size_t>(i)]);
    half2_as_float[static_cast<size_t>(i)] = __half2float(host_half2_out[static_cast<size_t>(i)]);
  }
  const ErrorStats half_error = compare_float_vectors(host_float_out, half_as_float);
  const ErrorStats half2_error = compare_float_vectors(host_float_out, half2_as_float);

  std::cout << "experiment: half2_arithmetic\n";
  std::cout << "variant kernel_ms speedup_vs_float max_abs_error max_rel_error mean_rel_error\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "float " << float_time.ms << " " << 1.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << '\n';
  std::cout << "half " << half_time.ms << " " << (float_time.ms / half_time.ms) << " "
            << half_error.max_abs_error << " " << half_error.max_rel_error << " " << half_error.mean_rel_error << '\n';
  std::cout << "half2 " << half2_time.ms << " " << (float_time.ms / half2_time.ms) << " "
            << half2_error.max_abs_error << " " << half2_error.max_rel_error << " " << half2_error.mean_rel_error << '\n';

  CHECK_CUDA(cudaFree(d_float_in));
  CHECK_CUDA(cudaFree(d_float_out));
  CHECK_CUDA(cudaFree(d_half_in));
  CHECK_CUDA(cudaFree(d_half_out));
  CHECK_CUDA(cudaFree(d_half2_in));
  CHECK_CUDA(cudaFree(d_half2_out));
}

void run_divmod_experiment(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  int *d_q_div = nullptr, *d_q_shift = nullptr, *d_r_mod = nullptr, *d_r_mask = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q_div, static_cast<size_t>(kDivElems) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_q_shift, static_cast<size_t>(kDivElems) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_r_mod, static_cast<size_t>(kDivElems) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_r_mask, static_cast<size_t>(kDivElems) * sizeof(int)));

  const int grid = (kDivElems + kBlockSize - 1) / kBlockSize;
  div_kernel<<<grid, kBlockSize, 0, stream>>>(d_q_div, kDivElems, kDivPow2);
  shift_kernel<<<grid, kBlockSize, 0, stream>>>(d_q_shift, kDivElems, kDivShift);
  mod_kernel<<<grid, kBlockSize, 0, stream>>>(d_r_mod, kDivElems, kDivPow2);
  mask_kernel<<<grid, kBlockSize, 0, stream>>>(d_r_mask, kDivElems, kDivPow2 - 1);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start, stream));
  div_kernel<<<grid, kBlockSize, 0, stream>>>(d_q_div, kDivElems, kDivPow2);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult div_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  shift_kernel<<<grid, kBlockSize, 0, stream>>>(d_q_shift, kDivElems, kDivShift);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult shift_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  mod_kernel<<<grid, kBlockSize, 0, stream>>>(d_r_mod, kDivElems, kDivPow2);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult mod_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  mask_kernel<<<grid, kBlockSize, 0, stream>>>(d_r_mask, kDivElems, kDivPow2 - 1);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult mask_time = measure_events(stream, start, stop);

  std::vector<int> host_q_div(static_cast<size_t>(kDivElems));
  std::vector<int> host_q_shift(static_cast<size_t>(kDivElems));
  std::vector<int> host_r_mod(static_cast<size_t>(kDivElems));
  std::vector<int> host_r_mask(static_cast<size_t>(kDivElems));
  CHECK_CUDA(cudaMemcpy(host_q_div.data(), d_q_div, static_cast<size_t>(kDivElems) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_q_shift.data(), d_q_shift, static_cast<size_t>(kDivElems) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_r_mod.data(), d_r_mod, static_cast<size_t>(kDivElems) * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_r_mask.data(), d_r_mask, static_cast<size_t>(kDivElems) * sizeof(int), cudaMemcpyDeviceToHost));

  const bool q_ok = validate_int_vectors(host_q_div, host_q_shift);
  const bool r_ok = validate_int_vectors(host_r_mod, host_r_mask);
  std::cout << "experiment: divmod_strength_reduction\n";
  std::cout << "variant kernel_ms speedup_vs_baseline check\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "div " << div_time.ms << " " << 1.0 << " PASS\n";
  std::cout << "shift " << shift_time.ms << " " << (div_time.ms / shift_time.ms) << " "
            << (q_ok ? "PASS" : "FAIL") << '\n';
  std::cout << "mod " << mod_time.ms << " " << 1.0 << " PASS\n";
  std::cout << "mask " << mask_time.ms << " " << (mod_time.ms / mask_time.ms) << " "
            << (r_ok ? "PASS" : "FAIL") << '\n';

  CHECK_CUDA(cudaFree(d_q_div));
  CHECK_CUDA(cudaFree(d_q_shift));
  CHECK_CUDA(cudaFree(d_r_mod));
  CHECK_CUDA(cudaFree(d_r_mask));
}

void run_fdividef_experiment(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  std::vector<float> host_x(static_cast<size_t>(kFdivElems));
  std::vector<float> host_y(static_cast<size_t>(kFdivElems));
  for (int i = 0; i < kFdivElems; ++i) {
    host_x[static_cast<size_t>(i)] = 0.5f + static_cast<float>(i % 2048) * 0.001f;
    host_y[static_cast<size_t>(i)] = 1.0f + static_cast<float>((i * 3) % 1024) * 0.002f;
  }

  float *d_x = nullptr, *d_y = nullptr, *d_precise = nullptr, *d_fast = nullptr, *d_rcp = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(kFdivElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, static_cast<size_t>(kFdivElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_precise, static_cast<size_t>(kFdivElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_fast, static_cast<size_t>(kFdivElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_rcp, static_cast<size_t>(kFdivElems) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, host_x.data(), static_cast<size_t>(kFdivElems) * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, host_y.data(), static_cast<size_t>(kFdivElems) * sizeof(float), cudaMemcpyHostToDevice));

  const int grid = (kFdivElems + kBlockSize - 1) / kBlockSize;
  fdiv_precise_kernel<<<grid, kBlockSize, 0, stream>>>(d_precise, d_x, d_y, kFdivElems, kRepeats);
  fdiv_fast_kernel<<<grid, kBlockSize, 0, stream>>>(d_fast, d_x, d_y, kFdivElems, kRepeats);
  fdiv_rcp_kernel<<<grid, kBlockSize, 0, stream>>>(d_rcp, d_x, d_y, kFdivElems, kRepeats);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start, stream));
  fdiv_precise_kernel<<<grid, kBlockSize, 0, stream>>>(d_precise, d_x, d_y, kFdivElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult precise_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  fdiv_fast_kernel<<<grid, kBlockSize, 0, stream>>>(d_fast, d_x, d_y, kFdivElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult fast_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  fdiv_rcp_kernel<<<grid, kBlockSize, 0, stream>>>(d_rcp, d_x, d_y, kFdivElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult rcp_time = measure_events(stream, start, stop);

  std::vector<float> host_precise(static_cast<size_t>(kFdivElems));
  std::vector<float> host_fast(static_cast<size_t>(kFdivElems));
  std::vector<float> host_rcp(static_cast<size_t>(kFdivElems));
  CHECK_CUDA(cudaMemcpy(host_precise.data(), d_precise, static_cast<size_t>(kFdivElems) * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_fast.data(), d_fast, static_cast<size_t>(kFdivElems) * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_rcp.data(), d_rcp, static_cast<size_t>(kFdivElems) * sizeof(float), cudaMemcpyDeviceToHost));
  const ErrorStats error = compare_float_vectors(host_precise, host_fast);
  const ErrorStats rcp_error = compare_float_vectors(host_precise, host_rcp);

  std::cout << "experiment: fdividef_fast_math\n";
  std::cout << "variant kernel_ms speedup_vs_precise max_abs_error max_rel_error mean_rel_error\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "precise " << precise_time.ms << " " << 1.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << '\n';
  std::cout << "fdividef " << fast_time.ms << " " << (precise_time.ms / fast_time.ms) << " "
            << error.max_abs_error << " " << error.max_rel_error << " " << error.mean_rel_error << '\n';
  std::cout << "frcp_mul " << rcp_time.ms << " " << (precise_time.ms / rcp_time.ms) << " "
            << rcp_error.max_abs_error << " " << rcp_error.max_rel_error << " " << rcp_error.mean_rel_error << '\n';

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_precise));
  CHECK_CUDA(cudaFree(d_fast));
  CHECK_CUDA(cudaFree(d_rcp));
}

void run_rsqrt_experiment(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  std::vector<float> host_in(static_cast<size_t>(kRsqrtElems));
  for (int i = 0; i < kRsqrtElems; ++i) {
    host_in[static_cast<size_t>(i)] = 1.0e-4f + static_cast<float>(i % 10000) * 1.0e-3f;
  }

  float *d_in = nullptr, *d_precise = nullptr, *d_fast = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, static_cast<size_t>(kRsqrtElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_precise, static_cast<size_t>(kRsqrtElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_fast, static_cast<size_t>(kRsqrtElems) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_in, host_in.data(), static_cast<size_t>(kRsqrtElems) * sizeof(float), cudaMemcpyHostToDevice));

  const int grid = (kRsqrtElems + kBlockSize - 1) / kBlockSize;
  rsqrt_precise_kernel<<<grid, kBlockSize, 0, stream>>>(d_precise, d_in, kRsqrtElems, kRepeats);
  rsqrt_fast_kernel<<<grid, kBlockSize, 0, stream>>>(d_fast, d_in, kRsqrtElems, kRepeats);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start, stream));
  rsqrt_precise_kernel<<<grid, kBlockSize, 0, stream>>>(d_precise, d_in, kRsqrtElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult precise_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  rsqrt_fast_kernel<<<grid, kBlockSize, 0, stream>>>(d_fast, d_in, kRsqrtElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult fast_time = measure_events(stream, start, stop);

  std::vector<float> host_precise(static_cast<size_t>(kRsqrtElems));
  std::vector<float> host_fast(static_cast<size_t>(kRsqrtElems));
  CHECK_CUDA(cudaMemcpy(host_precise.data(), d_precise, static_cast<size_t>(kRsqrtElems) * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_fast.data(), d_fast, static_cast<size_t>(kRsqrtElems) * sizeof(float), cudaMemcpyDeviceToHost));
  const ErrorStats fast_error = compare_float_vectors(host_precise, host_fast);

  std::cout << "experiment: rsqrt_fast_math\n";
  std::cout << "variant kernel_ms speedup_vs_precise max_abs_error max_rel_error mean_rel_error\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "precise " << precise_time.ms << " " << 1.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << '\n';
  std::cout << "fast_rsqrt " << fast_time.ms << " " << (precise_time.ms / fast_time.ms) << " "
            << fast_error.max_abs_error << " " << fast_error.max_rel_error << " " << fast_error.mean_rel_error << '\n';

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_precise));
  CHECK_CUDA(cudaFree(d_fast));
}

void run_normalize3_experiment(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  std::vector<float> host_x(static_cast<size_t>(kNormElems));
  std::vector<float> host_y(static_cast<size_t>(kNormElems));
  std::vector<float> host_z(static_cast<size_t>(kNormElems));
  for (int i = 0; i < kNormElems; ++i) {
    host_x[static_cast<size_t>(i)] = 0.1f + static_cast<float>(i % 1024) * 1.0e-3f;
    host_y[static_cast<size_t>(i)] = 0.2f + static_cast<float>((i * 3) % 1024) * 1.0e-3f;
    host_z[static_cast<size_t>(i)] = 0.3f + static_cast<float>((i * 7) % 1024) * 1.0e-3f;
  }

  float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_precise = nullptr, *d_fast = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(kNormElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, static_cast<size_t>(kNormElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_z, static_cast<size_t>(kNormElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_precise, static_cast<size_t>(kNormElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_fast, static_cast<size_t>(kNormElems) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, host_x.data(), static_cast<size_t>(kNormElems) * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, host_y.data(), static_cast<size_t>(kNormElems) * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_z, host_z.data(), static_cast<size_t>(kNormElems) * sizeof(float), cudaMemcpyHostToDevice));

  const int grid = (kNormElems + kBlockSize - 1) / kBlockSize;
  normalize3_precise_kernel<<<grid, kBlockSize, 0, stream>>>(d_precise, d_x, d_y, d_z, kNormElems, kRepeats);
  normalize3_rsqrt_kernel<<<grid, kBlockSize, 0, stream>>>(d_fast, d_x, d_y, d_z, kNormElems, kRepeats);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start, stream));
  normalize3_precise_kernel<<<grid, kBlockSize, 0, stream>>>(d_precise, d_x, d_y, d_z, kNormElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult precise_time = measure_events(stream, start, stop);

  CHECK_CUDA(cudaEventRecord(start, stream));
  normalize3_rsqrt_kernel<<<grid, kBlockSize, 0, stream>>>(d_fast, d_x, d_y, d_z, kNormElems, kRepeats);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  const TimingResult fast_time = measure_events(stream, start, stop);

  std::vector<float> host_precise(static_cast<size_t>(kNormElems));
  std::vector<float> host_fast(static_cast<size_t>(kNormElems));
  CHECK_CUDA(cudaMemcpy(host_precise.data(), d_precise, static_cast<size_t>(kNormElems) * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_fast.data(), d_fast, static_cast<size_t>(kNormElems) * sizeof(float), cudaMemcpyDeviceToHost));
  const ErrorStats error = compare_float_vectors(host_precise, host_fast);

  std::cout << "experiment: normalize3_rsqrt\n";
  std::cout << "variant kernel_ms speedup_vs_precise max_abs_error max_rel_error mean_rel_error\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "precise " << precise_time.ms << " " << 1.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << '\n';
  std::cout << "rsqrt " << fast_time.ms << " " << (precise_time.ms / fast_time.ms) << " "
            << error.max_abs_error << " " << error.max_rel_error << " " << error.mean_rel_error << '\n';

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_z));
  CHECK_CUDA(cudaFree(d_precise));
  CHECK_CUDA(cudaFree(d_fast));
}

}  // namespace

int main() {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  run_half2_experiment(stream, start, stop);
  run_divmod_experiment(stream, start, stop);
  run_fdividef_experiment(stream, start, stop);
  run_rsqrt_experiment(stream, start, stop);
  run_normalize3_experiment(stream, start, stop);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
