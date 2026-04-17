#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef ONLINE_JUDGE
#include <iostream>
#include <string>
#endif

constexpr int kDModel = 768;
constexpr int kNumHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kFfnDim = 3072;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kApproxCoeff = 0.044715f;
constexpr size_t kGamma1Offset = 0;
constexpr size_t kBeta1Offset = 768;
constexpr size_t kWqkvOffset = 1536;
constexpr size_t kBqkvOffset = 1771008;
constexpr size_t kWAttnOffset = 1773312;
constexpr size_t kBAttnOffset = 2363136;
constexpr size_t kGamma2Offset = 2363904;
constexpr size_t kBeta2Offset = 2364672;
constexpr size_t kWfcOffset = 2365440;
constexpr size_t kBfcOffset = 4724736;
constexpr size_t kWProjOffset = 4727808;
constexpr size_t kBProjOffset = 7087104;
constexpr size_t kWeightsSize = 7087872;
constexpr float kCompareEps = 2e-4f;

__global__ void layerNorm(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len || threadIdx.x != 0) {
    return;
  }

  const float* gamma = weights + kGamma1Offset;
  const float* beta = weights + kBeta1Offset;
  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  float* output_row = output + static_cast<size_t>(row) * kDModel;

  float mean = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    mean += input_row[i];
  }
  mean /= static_cast<float>(kDModel);

  float var = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    const float diff = input_row[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(kDModel);

  const float inv_std = rsqrtf(var + kLayerNormEps);
  for (int i = 0; i < kDModel; ++i) {
    output_row[i] = ((input_row[i] - mean) * inv_std) * gamma[i] + beta[i];
  }
}

__global__ void layerNorm2(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len || threadIdx.x != 0) {
    return;
  }

  const float* gamma = weights + kGamma2Offset;
  const float* beta = weights + kBeta2Offset;
  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  float* output_row = output + static_cast<size_t>(row) * kDModel;

  float mean = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    mean += input_row[i];
  }
  mean /= static_cast<float>(kDModel);

  float var = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    const float diff = input_row[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(kDModel);

  const float inv_std = rsqrtf(var + kLayerNormEps);
  for (int i = 0; i < kDModel; ++i) {
    output_row[i] = ((input_row[i] - mean) * inv_std) * gamma[i] + beta[i];
  }
}

__global__ void qkv(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len) {
    return;
  }

  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  const float* w_qkv = weights + kWqkvOffset;
  const float* b_qkv = weights + kBqkvOffset;
  float* output_row = output + static_cast<size_t>(row) * (kDModel * 3);

  for (int out_col = threadIdx.x; out_col < kDModel * 3; out_col += blockDim.x) {
    float sum = b_qkv[out_col];
    for (int in_col = 0; in_col < kDModel; ++in_col) {
      sum += input_row[in_col] * w_qkv[in_col * (kDModel * 3) + out_col];
    }
    output_row[out_col] = sum;
  }
}

__global__ void attn(const float* qkv, float* output, const float* weights, int seq_len) {
  (void)weights;
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= seq_len || head >= kNumHeads || lane >= kHeadDim) {
    return;
  }

  extern __shared__ float scores[];

  const int row_base = row * (kDModel * 3);
  const int q_base = row_base + head * kHeadDim;

  if (lane == 0) {
    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
      const int k_base = j * (kDModel * 3) + kDModel + head * kHeadDim;
      float dot = 0.0f;
      for (int d = 0; d < kHeadDim; ++d) {
        dot += qkv[q_base + d] * qkv[k_base + d];
      }
      const float score = dot / sqrtf(static_cast<float>(kHeadDim));
      scores[j] = score;
      max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
      scores[j] = expf(scores[j] - max_score);
      denom += scores[j];
    }

    const float inv_denom = 1.0f / denom;
    for (int j = 0; j < seq_len; ++j) {
      scores[j] *= inv_denom;
    }
  }
  __syncthreads();

  float acc = 0.0f;
  for (int j = 0; j < seq_len; ++j) {
    const int v_base = j * (kDModel * 3) + 2 * kDModel + head * kHeadDim;
    acc += scores[j] * qkv[v_base + lane];
  }
  output[row * kDModel + head * kHeadDim + lane] = acc;
}

__global__ void linear_bias(const float* input,
                            float* output,
                            const float* weight,
                            const float* bias,
                            int rows,
                            int in_dim,
                            int out_dim) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= out_dim) {
    return;
  }

  float sum = bias != nullptr ? bias[col] : 0.0f;
  const float* input_row = input + static_cast<size_t>(row) * in_dim;
  for (int k = 0; k < in_dim; ++k) {
    sum += input_row[k] * weight[k * out_dim + col];
  }
  output[static_cast<size_t>(row) * out_dim + col] = sum;
}

__device__ float gelu(float x) {
  const float cubic = x * x * x;
  const float inner = kSqrt2OverPi * (x + kApproxCoeff * cubic);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void ffn(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len) {
    return;
  }

  __shared__ float up[kFfnDim];

  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  const float* wfc = weights + kWfcOffset;
  const float* bfc = weights + kBfcOffset;
  const float* wproj = weights + kWProjOffset;
  const float* bproj = weights + kBProjOffset;
  float* output_row = output + static_cast<size_t>(row) * kDModel;

  for (int hidden = threadIdx.x; hidden < kFfnDim; hidden += blockDim.x) {
    float sum = bfc[hidden];
    for (int i = 0; i < kDModel; ++i) {
      sum += input_row[i] * wfc[i * kFfnDim + hidden];
    }
    up[hidden] = gelu(sum);
  }
  __syncthreads();

  for (int out_col = threadIdx.x; out_col < kDModel; out_col += blockDim.x) {
    float sum = bproj[out_col];
    for (int hidden = 0; hidden < kFfnDim; ++hidden) {
      sum += up[hidden] * wproj[hidden * kDModel + out_col];
    }
    output_row[out_col] = sum;
  }
}

__global__ void add_residual(const float* a, const float* b, float* output, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = a[idx] + b[idx];
  }
}

namespace {

std::vector<float> gpt2_block_reference(const std::vector<float>& x,
                                        const std::vector<float>& weights,
                                        int seq_len);

}  // namespace

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
  if (x == nullptr || output == nullptr || weights == nullptr || seq_len <= 0) {
    return;
  }

  const size_t token_count = static_cast<size_t>(seq_len) * kDModel;
  const size_t qkv_count = static_cast<size_t>(seq_len) * kDModel * 3;
  const size_t bytes = token_count * sizeof(float);
  const size_t qkv_bytes = qkv_count * sizeof(float);

  float* ln1 = nullptr;
  float* qkv_buf = nullptr;
  float* attn_concat = nullptr;
  float* attn_proj = nullptr;
  float* residual1 = nullptr;
  float* ln2 = nullptr;
  float* ff2 = nullptr;

  auto cleanup = [&]() {
    if (ln1 != nullptr) cudaFree(ln1);
    if (qkv_buf != nullptr) cudaFree(qkv_buf);
    if (attn_concat != nullptr) cudaFree(attn_concat);
    if (attn_proj != nullptr) cudaFree(attn_proj);
    if (residual1 != nullptr) cudaFree(residual1);
    if (ln2 != nullptr) cudaFree(ln2);
    if (ff2 != nullptr) cudaFree(ff2);
  };

  if (cudaMalloc(&ln1, bytes) != cudaSuccess ||
      cudaMalloc(&qkv_buf, qkv_bytes) != cudaSuccess ||
      cudaMalloc(&attn_concat, bytes) != cudaSuccess ||
      cudaMalloc(&attn_proj, bytes) != cudaSuccess ||
      cudaMalloc(&residual1, bytes) != cudaSuccess ||
      cudaMalloc(&ln2, bytes) != cudaSuccess ||
      cudaMalloc(&ff2, bytes) != cudaSuccess) {
    cleanup();
    return;
  }

  layerNorm<<<seq_len, 1>>>(x, ln1, weights, seq_len);
  qkv<<<seq_len, 256>>>(ln1, qkv_buf, weights, seq_len);
  attn<<<dim3(seq_len, kNumHeads), kHeadDim, static_cast<size_t>(seq_len) * sizeof(float)>>>(
      qkv_buf, attn_concat, weights, seq_len);

  const dim3 linear_grid((kDModel + 255) / 256, seq_len);
  linear_bias<<<linear_grid, 256>>>(
      attn_concat,
      attn_proj,
      weights + kWAttnOffset,
      weights + kBAttnOffset,
      seq_len,
      kDModel,
      kDModel);

  add_residual<<<static_cast<int>((token_count + 255) / 256), 256>>>(x, attn_proj, residual1,
                                                                      static_cast<int>(token_count));

  layerNorm2<<<seq_len, 1>>>(residual1, ln2, weights, seq_len);
  ffn<<<seq_len, 256>>>(ln2, ff2, weights, seq_len);

  add_residual<<<static_cast<int>((token_count + 255) / 256), 256>>>(residual1, ff2, output,
                                                                      static_cast<int>(token_count));

  cleanup();
}

#ifndef ONLINE_JUDGE
namespace {

float gelu_tanh(float x) {
  const float cubic = x * x * x;
  const float inner = kSqrt2OverPi * (x + kApproxCoeff * cubic);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kCompareEps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

std::vector<float> make_patterned_data(size_t n, int period, float scale, float bias) {
  std::vector<float> values(n);
  for (size_t i = 0; i < n; ++i) {
    const int centered = static_cast<int>(i % static_cast<size_t>(period)) - (period / 2);
    values[i] = centered * scale + bias;
  }
  return values;
}

void layer_norm_row(const float* input,
                    const float* gamma,
                    const float* beta,
                    float* output) {
  float mean = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    mean += input[i];
  }
  mean /= static_cast<float>(kDModel);

  float var = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    const float diff = input[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(kDModel);

  const float inv_std = 1.0f / std::sqrt(var + kLayerNormEps);
  for (int i = 0; i < kDModel; ++i) {
    output[i] = ((input[i] - mean) * inv_std) * gamma[i] + beta[i];
  }
}

void matmul_add_bias(const float* input,
                     int rows,
                     int in_dim,
                     const float* weights,
                     int out_dim,
                     const float* bias,
                     float* output) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < out_dim; ++c) {
      float sum = bias != nullptr ? bias[c] : 0.0f;
      for (int k = 0; k < in_dim; ++k) {
        sum += input[r * in_dim + k] * weights[k * out_dim + c];
      }
      output[r * out_dim + c] = sum;
    }
  }
}

std::vector<float> gpt2_block_reference(const std::vector<float>& x,
                                        const std::vector<float>& weights,
                                        int seq_len) {
  const float* gamma1 = weights.data() + kGamma1Offset;
  const float* beta1 = weights.data() + kBeta1Offset;
  const float* wqkv = weights.data() + kWqkvOffset;
  const float* bqkv = weights.data() + kBqkvOffset;
  const float* wattn = weights.data() + kWAttnOffset;
  const float* battn = weights.data() + kBAttnOffset;
  const float* gamma2 = weights.data() + kGamma2Offset;
  const float* beta2 = weights.data() + kBeta2Offset;
  const float* wfc = weights.data() + kWfcOffset;
  const float* bfc = weights.data() + kBfcOffset;
  const float* wproj = weights.data() + kWProjOffset;
  const float* bproj = weights.data() + kBProjOffset;

  std::vector<float> ln1(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (int row = 0; row < seq_len; ++row) {
    layer_norm_row(x.data() + row * kDModel, gamma1, beta1, ln1.data() + row * kDModel);
  }

  std::vector<float> qkv(static_cast<size_t>(seq_len) * kDModel * 3, 0.0f);
  matmul_add_bias(ln1.data(), seq_len, kDModel, wqkv, kDModel * 3, bqkv, qkv.data());

  std::vector<float> attn_concat(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<float> scores(seq_len * seq_len, 0.0f);
  std::vector<float> weights_softmax(seq_len * seq_len, 0.0f);

  for (int head = 0; head < kNumHeads; ++head) {
    for (int i = 0; i < seq_len; ++i) {
      float max_score = -INFINITY;
      for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
          const int q_idx = i * (kDModel * 3) + head * kHeadDim + d;
          const int k_idx = j * (kDModel * 3) + kDModel + head * kHeadDim + d;
          dot += qkv[q_idx] * qkv[k_idx];
        }
        const float score = dot * scale;
        scores[i * seq_len + j] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        const float weight = std::exp(scores[i * seq_len + j] - max_score);
        weights_softmax[i * seq_len + j] = weight;
        denom += weight;
      }

      for (int d = 0; d < kHeadDim; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
          const int v_idx = j * (kDModel * 3) + 2 * kDModel + head * kHeadDim + d;
          acc += (weights_softmax[i * seq_len + j] / denom) * qkv[v_idx];
        }
        attn_concat[i * kDModel + head * kHeadDim + d] = acc;
      }
    }
  }

  std::vector<float> attn_proj(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  matmul_add_bias(attn_concat.data(), seq_len, kDModel, wattn, kDModel, battn, attn_proj.data());

  std::vector<float> residual1(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (size_t i = 0; i < residual1.size(); ++i) {
    residual1[i] = x[i] + attn_proj[i];
  }

  std::vector<float> ln2(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (int row = 0; row < seq_len; ++row) {
    layer_norm_row(residual1.data() + row * kDModel, gamma2, beta2, ln2.data() + row * kDModel);
  }

  std::vector<float> ff1(static_cast<size_t>(seq_len) * kFfnDim, 0.0f);
  matmul_add_bias(ln2.data(), seq_len, kDModel, wfc, kFfnDim, bfc, ff1.data());
  for (float& value : ff1) {
    value = gelu_tanh(value);
  }

  std::vector<float> ff2(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  matmul_add_bias(ff1.data(), seq_len, kFfnDim, wproj, kDModel, bproj, ff2.data());

  std::vector<float> output(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = residual1[i] + ff2[i];
  }
  return output;
}

bool run_case(const std::string& name,
              const std::vector<float>& x,
              const std::vector<float>& weights,
              int seq_len) {
  const size_t x_bytes = x.size() * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(seq_len) * kDModel * sizeof(float);
  const size_t weights_bytes = weights.size() * sizeof(float);
  const std::vector<float> expected = gpt2_block_reference(x, weights, seq_len);
  std::vector<float> actual(expected.size(), 0.0f);

  float* d_x = nullptr;
  float* d_out = nullptr;
  float* d_weights = nullptr;
  auto cleanup = [&]() {
    if (d_x != nullptr) cudaFree(d_x);
    if (d_out != nullptr) cudaFree(d_out);
    if (d_weights != nullptr) cudaFree(d_weights);
  };

  if (cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
      cudaMalloc(&d_out, out_bytes) != cudaSuccess ||
      cudaMalloc(&d_weights, weights_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_x, x.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_weights, weights.data(), weights_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemset(d_out, 0, out_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": device upload failed\n";
    cleanup();
    return false;
  }

  solve(d_x, d_out, d_weights, seq_len);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution failed\n";
    cleanup();
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!almost_equal(actual[i], expected[i])) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << i
                << ", expected=" << expected[i]
                << ", actual=" << actual[i] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

std::vector<float> make_weights_pattern(float scale) {
  std::vector<float> weights(kWeightsSize, 0.0f);
  for (size_t i = 0; i < weights.size(); ++i) {
    const int centered = static_cast<int>(i % 17) - 8;
    weights[i] = centered * scale;
  }
  return weights;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int seq_len = 1;
    const std::vector<float> x =
        make_patterned_data(static_cast<size_t>(seq_len) * kDModel, 19, 0.01f, -0.02f);
    const std::vector<float> weights = make_weights_pattern(1e-4f);
    passed += run_case("single_token_reference", x, weights, seq_len) ? 1 : 0;
  }

  {
    ++total;
    const int seq_len = 2;
    const std::vector<float> x =
        make_patterned_data(static_cast<size_t>(seq_len) * kDModel, 23, 0.02f, 0.01f);
    const std::vector<float> weights = make_weights_pattern(8e-5f);
    passed += run_case("two_token_reference", x, weights, seq_len) ? 1 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
  return passed == total ? 0 : 1;
}
#endif
