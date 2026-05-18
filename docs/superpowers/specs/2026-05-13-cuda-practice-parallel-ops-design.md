# CUDA Practice Parallel Operators Design

## Goal

Extend the CUDA practice workspace with exercises `16` through `20`, moving into classic parallel algorithms and matrix/image operators that build practical CUDA fluency.

## Exercise Set

The new exercises are:

1. `16-prefix-sum`: one-block inclusive scan using shared memory.
2. `17-matrix-vector`: row-wise matrix-vector multiplication.
3. `18-tiled-matmul`: shared-memory tiled matrix multiplication.
4. `19-conv2d`: single-channel `3x3` convolution with zero padding.
5. `20-max-pooling`: `2x2` max pooling with stride `2`.

## Student Experience

Each exercise follows the existing `practice/` style:

- One numbered directory with `README.md` and `exercise.cu`.
- A single marked CUDA kernel.
- `constexpr bool kStudentKernelImplemented = false` until the learner is ready to validate.
- CPU reference implementation, correctness check, and CUDA event timing.

## Learning Progression

`16-prefix-sum` introduces scan, which is a new parallel primitive beyond reductions. `17-matrix-vector` applies row-wise reductions to a linear algebra operator. `18-tiled-matmul` is the main CUDA shared-memory milestone. `19-conv2d` introduces stencil-style neighborhood access. `20-max-pooling` reinforces output-coordinate mapping for downsampled image operators.

## Verification

The workspace should continue to configure and build with:

```bash
cmake -S practice -B practice/build
cmake --build practice/build -j
```

Before implementation, each new exercise should print TODO guidance and exit with code `2`.
