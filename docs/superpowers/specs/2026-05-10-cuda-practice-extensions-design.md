# CUDA Practice Extensions Design

## Goal

Extend the beginner CUDA practice workspace with five exercises after `05-softmax`, giving the learner a bridge from indexing and block reductions into common transformer operators and practical optimization patterns.

## Exercise Set

The new exercises are:

1. `06-rmsnorm`: row-wise RMSNorm with scale weights.
2. `07-gelu`: elementwise GELU activation using the tanh approximation.
3. `08-rope`: rotary positional embedding over token/head pairs.
4. `09-optimized-reduction`: a faster sum reduction where each thread accumulates multiple elements before block reduction.
5. `10-tiled-transpose`: shared-memory tiled matrix transpose with padding to reduce shared-memory bank conflicts.

## Student Experience

Each exercise follows the existing `practice/` pattern:

- A numbered directory with `README.md` and `exercise.cu`.
- A single marked kernel for the learner to implement.
- `constexpr bool kStudentKernelImplemented = false` until the learner is ready to check their solution.
- CPU reference implementation, correctness check, and `cudaEvent` timing.
- Initial executable behavior is a clear TODO message.

## Learning Progression

`06-rmsnorm` reuses reduction ideas from softmax while introducing normalization. `07-gelu` gives a low-friction elementwise math exercise. `08-rope` introduces the indexing shape common in attention code. `09-optimized-reduction` revisits reduction from a performance angle. `10-tiled-transpose` revisits transpose with shared memory and coalescing.

## Verification

The workspace should configure and build with the same commands:

```bash
cmake -S practice -B practice/build
cmake --build practice/build -j
```

Running each new exercise before implementation should exit with TODO guidance. After the learner implements a kernel and flips the boolean, the exercise should compare against the CPU reference and print timing.
