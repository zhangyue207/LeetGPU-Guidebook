# CUDA Practice Workspace

This folder is your kernel-writing area. The original guidebook docs and completed solutions stay unchanged; practice code lives here.

## Build

From the repository root:

```bash
cmake -S practice -B practice/build
cmake --build practice/build -j
```

If CMake cannot infer your GPU architecture, pass one explicitly:

```bash
cmake -S practice -B practice/build -DCMAKE_CUDA_ARCHITECTURES=80
```

## Check CUDA First

```bash
./practice/build/00-sanity-check/cuda_sanity
```

This executable is already solved. It only checks that the CUDA compiler, runtime, launch path, and memory copy path work.

## Start Practicing

Open the exercise file, implement the marked kernel, then set:

```cpp
constexpr bool kStudentKernelImplemented = true;
```

Rebuild and run the exercise:

```bash
cmake --build practice/build -j
./practice/build/01-vector-add/vector_add
```

## Learning Order

1. `01-vector-add`: one-dimensional indexing.
2. `02-matrix-copy`: two-dimensional indexing.
3. `03-matrix-transpose`: row/column mapping and memory access.
4. `04-reduction`: shared memory and block partial sums.
5. `05-softmax`: stable row-wise softmax.
6. `06-rmsnorm`: row-wise normalization using a reduction.
7. `07-gelu`: elementwise activation math.
8. `08-rope`: rotary positional embedding indexing.
9. `09-optimized-reduction`: grid-stride accumulation plus block reduction.
10. `10-tiled-transpose`: shared-memory tiled transpose.
11. `11-saxpy`: fused elementwise multiply-add.
12. `12-layernorm`: row-wise normalization with mean and variance.
13. `13-vector-dot`: dot product with partial sums.
14. `14-histogram`: atomic updates into integer bins.
15. `15-embedding-lookup`: gather rows from an embedding table.
16. `16-prefix-sum`: one-block inclusive scan.
17. `17-matrix-vector`: row-wise matrix-vector multiplication.
18. `18-tiled-matmul`: shared-memory tiled matrix multiplication.
19. `19-conv2d`: single-channel 3x3 convolution.
20. `20-max-pooling`: 2x2 max pooling with stride 2.

Each exercise has a CPU reference, GPU output check, and `cudaEvent` kernel timing.
