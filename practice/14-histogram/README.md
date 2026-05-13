# 14 - Histogram

Implement a histogram over integer input values:

```cpp
histogram[x[i]] += 1
```

All values are in the range `[0, num_bins)`.

## What To Practice

- Integer kernels.
- Concurrent updates.
- `atomicAdd` for correctness when many threads update the same bin.

## Run

```bash
cmake --build practice/build -j
./practice/build/14-histogram/histogram
```

Edit `exercise.cu`, implement `histogram_kernel`, then set `kStudentKernelImplemented = true`.
