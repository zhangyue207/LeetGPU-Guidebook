# CUDA Practice Real Operators Design

## Goal

Extend the CUDA practice workspace with exercises `11` through `15`, moving from core indexing and reduction patterns into practical operators that appear in numerical computing and deep learning workloads.

## Exercise Set

The new exercises are:

1. `11-saxpy`: fused elementwise operation `y = alpha * x + y`.
2. `12-layernorm`: row-wise LayerNorm with mean and variance reductions.
3. `13-vector-dot`: vector dot product with partial sums.
4. `14-histogram`: integer histogram using `atomicAdd`.
5. `15-embedding-lookup`: gather rows from an embedding table by token id.

## Student Experience

Each exercise follows the existing workspace style:

- One numbered directory under `practice/`.
- A short `README.md`.
- A single `exercise.cu` with one marked kernel to implement.
- `constexpr bool kStudentKernelImplemented = false` until the learner is ready to validate.
- CPU reference, correctness check, and CUDA event timing.

## Learning Progression

`11-saxpy` keeps the learner warm with a fused elementwise operation. `12-layernorm` builds directly on RMSNorm by adding mean subtraction. `13-vector-dot` returns to reductions but changes the output shape to one scalar. `14-histogram` introduces safe concurrent updates through atomics. `15-embedding-lookup` introduces gather-style memory access common in language models.

## Verification

The workspace should continue to configure and build with:

```bash
cmake -S practice -B practice/build
cmake --build practice/build -j
```

Before implementation, each new exercise should print TODO guidance and exit with code `2`.
