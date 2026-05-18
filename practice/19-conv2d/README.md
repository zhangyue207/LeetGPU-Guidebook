# 19 - Conv2D

Implement a single-channel `3x3` convolution with zero padding:

```cpp
out[row, col] = sum(input[row + dy, col + dx] * kernel[dy, dx])
```

Out-of-bounds input coordinates contribute `0`.

## What To Practice

- Stencil-style neighborhood access.
- Boundary checks.
- Mapping one output element to one thread.

## Run

```bash
cmake --build practice/build -j
./practice/build/19-conv2d/conv2d
```

Edit `exercise.cu`, implement `conv2d_kernel`, then set `kStudentKernelImplemented = true`.
