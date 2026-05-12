# 09 - Optimized Reduction

Implement a faster block sum reduction. Compared with `04-reduction`, each thread should first accumulate multiple input elements using a grid-stride loop, then reduce one value per thread in shared memory.

The kernel still writes one partial sum per block:

```cpp
partial_sums[blockIdx.x] = block_sum
```

## What To Practice

- Grid-stride loops.
- Per-thread local accumulation.
- Shared-memory block reduction.
- Reducing the number of partial sums.

## Run

```bash
cmake --build practice/build -j
./practice/build/09-optimized-reduction/optimized_reduction
```

Edit `exercise.cu`, implement `optimized_sum_blocks_kernel`, then set `kStudentKernelImplemented = true`.
