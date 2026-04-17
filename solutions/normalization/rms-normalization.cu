#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace cg = cooperative_groups;

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kVecWidth = 4;
constexpr int kMaxFallbackBlocks = 128;

__device__ float g_sum_sq;
__device__ float g_scale;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int kThreads>
__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[kThreads / kWarpSize];

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    value = (threadIdx.x < kThreads / kWarpSize) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

template <int kThreads>
__launch_bounds__(kThreads)
__global__ void rms_cooperative_kernel(const float* __restrict__ input,
                                       float gamma,
                                       float beta,
                                       float* __restrict__ output,
                                       int N,
                                       float eps) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int vec_count = N / kVecWidth;

    if (blockIdx.x == 0 && tid == 0) {
        g_sum_sq = 0.0f;
    }
    grid.sync();

    float local_sum = 0.0f;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        const float value = input[idx];
        local_sum += value * value;
    }

    const float block_sum = block_reduce_sum<kThreads>(local_sum);
    if (tid == 0) {
        atomicAdd(&g_sum_sq, block_sum);
    }
    grid.sync();

    if (blockIdx.x == 0 && tid == 0) {
        g_scale = rsqrtf(g_sum_sq / static_cast<float>(N) + eps) * gamma;
    }
    grid.sync();

    const float scale = g_scale;
    float4* output4 = reinterpret_cast<float4*>(output);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        output4[vec_idx] = make_float4(
            fmaf(v.x, scale, beta),
            fmaf(v.y, scale, beta),
            fmaf(v.z, scale, beta),
            fmaf(v.w, scale, beta));
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        output[idx] = fmaf(input[idx], scale, beta);
    }
}

template <int kThreads>
__global__ void rms_sum_kernel(const float* __restrict__ input, int N) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_count = N / kVecWidth;

    float local_sum = 0.0f;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        const float value = input[idx];
        local_sum += value * value;
    }

    const float block_sum = block_reduce_sum<kThreads>(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&g_sum_sq, block_sum);
    }
}

__global__ void rms_finalize_scale_kernel(int N, float gamma, float eps) {
    g_scale = rsqrtf(g_sum_sq / static_cast<float>(N) + eps) * gamma;
}

template <int kThreads>
__global__ void rms_normalize_kernel(const float* __restrict__ input,
                                     float beta,
                                     float* __restrict__ output,
                                     int N) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_count = N / kVecWidth;
    const float scale = g_scale;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        output4[vec_idx] = make_float4(
            fmaf(v.x, scale, beta),
            fmaf(v.y, scale, beta),
            fmaf(v.z, scale, beta),
            fmaf(v.w, scale, beta));
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        output[idx] = fmaf(input[idx], scale, beta);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    if (N <= 0) {
        return;
    }

    int blocks = (N + kBlockSize - 1) / kBlockSize;
    if (blocks < 1) {
        blocks = 1;
    }

    int sm_count = 0;
    int cooperative_supported = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, 0);

    if (cooperative_supported) {
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, rms_cooperative_kernel<kBlockSize>, kBlockSize, 0);
        const int cooperative_limit = sm_count * max_blocks_per_sm;
        if (cooperative_limit > 0) {
            if (blocks > cooperative_limit) {
                blocks = cooperative_limit;
            }
            void* args[] = {const_cast<float**>(&input), &gamma, &beta, &output, &N, &eps};
            if (cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(rms_cooperative_kernel<kBlockSize>),
                    blocks,
                    kBlockSize,
                    args,
                    0,
                    nullptr) == cudaSuccess) {
                return;
            }
            cudaGetLastError();
        }
    }

    if (blocks > kMaxFallbackBlocks) {
        blocks = kMaxFallbackBlocks;
    }
    float zero = 0.0f;
    cudaMemcpyToSymbol(g_sum_sq, &zero, sizeof(float));
    rms_sum_kernel<kBlockSize><<<blocks, kBlockSize>>>(input, N);
    rms_finalize_scale_kernel<<<1, 1>>>(N, gamma, eps);
    rms_normalize_kernel<kBlockSize><<<blocks, kBlockSize>>>(input, beta, output, N);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-4f;
constexpr float kSentinel = 123456.0f;

std::vector<float> rms_reference(const std::vector<float>& input, float gamma, float beta,
                                 float eps) {
    const int n = static_cast<int>(input.size());
    float sum_sq = 0.0f;
    for (float v : input) {
        sum_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / n + eps);

    std::vector<float> expected(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        expected[i] = input[i] * inv_rms * gamma + beta;
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

bool run_case(const std::string& name, const std::vector<float>& input, float gamma, float beta,
              float eps) {
    const int n = static_cast<int>(input.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const std::vector<float> expected = rms_reference(input, gamma, beta, eps);
    std::vector<float> actual(n, 0.0f);
    const std::vector<float> sentinel(n, kSentinel);

    float* d_input = nullptr;
    float* d_output = nullptr;

    auto cleanup = [&]() {
        if (d_input != nullptr) cudaFree(d_input);
        if (d_output != nullptr) cudaFree(d_output);
    };

    if (cudaMalloc(&d_input, bytes) != cudaSuccess || cudaMalloc(&d_output, bytes) != cudaSuccess) {
        std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
        cleanup();
        return false;
    }

    bool ok = true;
    ok = ok && (cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    ok = ok && (cudaMemcpy(d_output, sentinel.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    if (!ok) {
        std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed\n";
        cleanup();
        return false;
    }

    solve(d_input, gamma, beta, d_output, n, eps);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "[FAIL] " << name << ": kernel execution or cudaMemcpy D2H failed\n";
        cleanup();
        return false;
    }

    bool unchanged = true;
    for (int i = 0; i < n; ++i) {
        if (!almost_equal(actual[i], kSentinel)) {
            unchanged = false;
            break;
        }
    }
    if (unchanged) {
        std::cerr << "[FAIL] " << name << ": output buffer was never updated by solve\n";
        cleanup();
        return false;
    }

    for (int i = 0; i < n; ++i) {
        if (!almost_equal(actual[i], expected[i])) {
            std::cerr << "[FAIL] " << name << ": mismatch at index " << i
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
        const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        passed += run_case("sample_case_n4", input, 1.0f, 0.0f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {1.0f, 2.0f, 3.0f};
        passed += run_case("sample_case_n3", input, 1.0f, 0.0f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {5.0f};
        passed += run_case("single_element_with_scale_and_shift", input, 2.0f, -1.5f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {0.0f, 0.0f, 0.0f, 0.0f};
        passed += run_case("all_zero_input", input, 1.25f, 3.0f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {-3.0f, -1.0f, 2.0f, 4.0f, -2.0f, 1.0f};
        passed += run_case("mixed_signs_nontrivial_affine", input, 0.75f, -0.25f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = make_patterned_data(257, 17, 0.5f, -1.0f);
        passed += run_case("patterned_large_input", input, 1.1f, 0.2f, 1e-5f) ? 1 : 0;
    }

    std::cout << "Summary: " << passed << "/" << total << " cases passed\n";
    return passed == total ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
