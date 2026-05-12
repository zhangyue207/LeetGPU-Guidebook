# 02 - Matrix Copy

Implement:

```cpp
out[row, col] = in[row, col]
```

The matrix is stored in row-major order:

```cpp
index = row * cols + col
```

## What To Practice

- Two-dimensional block and grid indexing.
- Mapping `(row, col)` to a linear offset.
- Bounds checks for partial edge blocks.

## Run

```bash
cmake --build practice/build -j
./practice/build/02-matrix-copy/matrix_copy
```

Edit `exercise.cu`, implement `matrix_copy_kernel`, then set `kStudentKernelImplemented = true`.

