# 13 - Vector Dot

Implement a dot product:

```cpp
result = sum(a[i] * b[i])
```

The kernel writes one partial dot product per block. The host sums partial results.

## What To Practice

- Multiplication followed by reduction.
- Grid-stride loops.
- Producing partial scalar outputs.

## Run

```bash
cmake --build practice/build -j
./practice/build/13-vector-dot/vector_dot
```

Edit `exercise.cu`, implement `dot_blocks_kernel`, then set `kStudentKernelImplemented = true`.
