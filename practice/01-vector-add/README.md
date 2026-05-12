# 01 - Vector Add

Implement:

```cpp
c[i] = a[i] + b[i]
```

## What To Practice

- One-dimensional global thread index.
- Bounds check for `i < n`.
- Choosing a reasonable block size.

## Run

```bash
cmake --build practice/build -j
./practice/build/01-vector-add/vector_add
```

Edit `exercise.cu`, implement `vector_add_kernel`, then set `kStudentKernelImplemented = true`.

