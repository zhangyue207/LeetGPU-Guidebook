# 04 - Reduction

Implement a block-level sum reduction. The kernel writes one partial sum per block:

```cpp
partial_sums[blockIdx.x] = sum(input elements covered by this block)
```

The host then sums the partial results and compares against a CPU reference.

## What To Practice

- Shared memory.
- `__syncthreads()`.
- Tree reduction inside one block.
- Handling arrays whose size is not exactly one block.

## Run

```bash
cmake --build practice/build -j
./practice/build/04-reduction/reduction
```

Edit `exercise.cu`, implement `sum_blocks_kernel`, then set `kStudentKernelImplemented = true`.

