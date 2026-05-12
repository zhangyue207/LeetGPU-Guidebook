# 05 - Softmax

Implement row-wise softmax:

```cpp
y[row, col] = exp(x[row, col] - max(row)) / sum(exp(x[row, k] - max(row)))
```

Each block handles one row. The exercise uses `cols = 256`, so one thread can handle one column.

## What To Practice

- Stable softmax using row max.
- Block-level max reduction.
- Block-level sum reduction.
- Reusing shared memory for multiple reductions.

## Run

```bash
cmake --build practice/build -j
./practice/build/05-softmax/softmax
```

Edit `exercise.cu`, implement `softmax_rows_kernel`, then set `kStudentKernelImplemented = true`.

