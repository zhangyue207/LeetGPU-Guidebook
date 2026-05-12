# 07 - GELU

Implement elementwise GELU using the common tanh approximation:

```cpp
y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2 / pi) * (x[i] + 0.044715 * x[i]^3)))
```

## What To Practice

- Elementwise CUDA kernels.
- Grid-size calculation for arbitrary vector lengths.
- Single-precision math functions such as `tanhf`.

## Run

```bash
cmake --build practice/build -j
./practice/build/07-gelu/gelu
```

Edit `exercise.cu`, implement `gelu_kernel`, then set `kStudentKernelImplemented = true`.
