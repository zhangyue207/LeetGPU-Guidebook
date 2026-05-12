# 08 - RoPE

Implement rotary positional embedding for a tensor shaped:

```text
tokens x heads x head_dim
```

For each pair of hidden dimensions:

```cpp
out_even = in_even * cos(theta) - in_odd * sin(theta)
out_odd  = in_even * sin(theta) + in_odd * cos(theta)
```

This exercise provides precomputed `cos` and `sin` tables with shape `tokens x (head_dim / 2)`.

## What To Practice

- Mapping one linear thread index to multiple tensor dimensions.
- Pairwise even/odd dimension transforms.
- Reading lookup tables.

## Run

```bash
cmake --build practice/build -j
./practice/build/08-rope/rope
```

Edit `exercise.cu`, implement `rope_kernel`, then set `kStudentKernelImplemented = true`.
