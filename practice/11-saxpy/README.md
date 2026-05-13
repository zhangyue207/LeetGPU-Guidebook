# 11 - SAXPY

Implement the fused elementwise operation:

```cpp
y[i] = alpha * x[i] + y[i]
```

This is the classic BLAS AXPY pattern with single-precision floats.

## What To Practice

- Elementwise kernels with one input and one in-place output.
- Reusing the one-dimensional indexing pattern.
- Fusing multiply and add work in one pass.

## Run

```bash
cmake --build practice/build -j
./practice/build/11-saxpy/saxpy
```

Edit `exercise.cu`, implement `saxpy_kernel`, then set `kStudentKernelImplemented = true`.
