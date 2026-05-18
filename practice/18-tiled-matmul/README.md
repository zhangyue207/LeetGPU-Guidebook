# 18 - Tiled Matrix Multiplication

Implement tiled matrix multiplication:

```cpp
C[row, col] = sum(A[row, k] * B[k, col])
```

All matrices are row-major.

## What To Practice

- Two-dimensional blocks.
- Shared-memory tiles for `A` and `B`.
- Synchronization between tile phases.
- Boundary checks for matrix edges.

## Run

```bash
cmake --build practice/build -j
./practice/build/18-tiled-matmul/tiled_matmul
```

Edit `exercise.cu`, implement `tiled_matmul_kernel`, then set `kStudentKernelImplemented = true`.
