# 03 - Matrix Transpose

Implement:

```cpp
out[col, row] = in[row, col]
```

Input shape is `rows x cols`; output shape is `cols x rows`.

## What To Practice

- Two-dimensional indexing.
- Thinking about input and output strides separately.
- First make the naive version correct. Then try a shared-memory tiled version.

## Run

```bash
cmake --build practice/build -j
./practice/build/03-matrix-transpose/matrix_transpose
```

Edit `exercise.cu`, implement `matrix_transpose_kernel`, then set `kStudentKernelImplemented = true`.

