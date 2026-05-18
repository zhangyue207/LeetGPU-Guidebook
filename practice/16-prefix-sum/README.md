# 16 - Prefix Sum

Implement an inclusive prefix sum for one block:

```cpp
y[i] = x[0] + x[1] + ... + x[i]
```

This exercise keeps the input length at one block so you can focus on the scan pattern before handling multi-block scans.

## What To Practice

- Shared-memory scan.
- `__syncthreads()` between scan strides.
- Difference between reduction and prefix sum.

## Run

```bash
cmake --build practice/build -j
./practice/build/16-prefix-sum/prefix_sum
```

Edit `exercise.cu`, implement `prefix_sum_kernel`, then set `kStudentKernelImplemented = true`.
