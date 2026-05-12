# 06 - RMSNorm

Implement row-wise RMSNorm:

```cpp
y[row, col] = x[row, col] * rsqrt(mean(x[row, :]^2) + eps) * weight[col]
```

The input is stored row-major with shape `rows x cols`.

## What To Practice

- One block per row.
- Squared-sum reduction in shared memory.
- `rsqrtf` for reciprocal square root.
- Reusing a per-row normalization value across threads.

## Run

```bash
cmake --build practice/build -j
./practice/build/06-rmsnorm/rmsnorm
```

Edit `exercise.cu`, implement `rmsnorm_rows_kernel`, then set `kStudentKernelImplemented = true`.
