# 12 - LayerNorm

Implement row-wise LayerNorm:

```cpp
mean = sum(x[row, :]) / cols
var = sum((x[row, :] - mean)^2) / cols
y[row, col] = (x[row, col] - mean) * rsqrt(var + eps) * gamma[col] + beta[col]
```

## What To Practice

- One block per row.
- Shared-memory reduction for the row mean.
- Shared-memory reduction for variance.
- Applying per-column scale and bias.

## Run

```bash
cmake --build practice/build -j
./practice/build/12-layernorm/layernorm
```

Edit `exercise.cu`, implement `layernorm_rows_kernel`, then set `kStudentKernelImplemented = true`.
