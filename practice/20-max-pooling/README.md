# 20 - Max Pooling

Implement `2x2` max pooling with stride `2`:

```cpp
output[out_row, out_col] = max(input[2*out_row:2*out_row+2, 2*out_col:2*out_col+2])
```

The input height and width are even.

## What To Practice

- Mapping output coordinates back to input windows.
- Downsampling.
- Small fixed-window reductions inside one thread.

## Run

```bash
cmake --build practice/build -j
./practice/build/20-max-pooling/max_pooling
```

Edit `exercise.cu`, implement `max_pooling_kernel`, then set `kStudentKernelImplemented = true`.
