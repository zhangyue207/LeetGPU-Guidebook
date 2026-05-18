# 17 - Matrix Vector Multiplication

Implement matrix-vector multiplication:

```cpp
y[row] = sum(A[row, col] * x[col])
```

The matrix is row-major with shape `rows x cols`.

## What To Practice

- One block per row.
- Row-wise dot product.
- Shared-memory reduction.

## Run

```bash
cmake --build practice/build -j
./practice/build/17-matrix-vector/matrix_vector
```

Edit `exercise.cu`, implement `matrix_vector_kernel`, then set `kStudentKernelImplemented = true`.
