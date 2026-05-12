# 10 - Tiled Transpose

Implement a shared-memory tiled matrix transpose:

```cpp
out[col, row] = in[row, col]
```

The starter declares a padded tile:

```cpp
__shared__ float tile[kTileDim][kTileDim + 1];
```

The extra column helps reduce shared-memory bank conflicts during the transposed read.

## What To Practice

- Loading a coalesced tile from global memory.
- Synchronizing before reading from shared memory.
- Writing the transposed tile back to global memory.
- Handling partial edge tiles.

## Run

```bash
cmake --build practice/build -j
./practice/build/10-tiled-transpose/tiled_transpose
```

Edit `exercise.cu`, implement `tiled_transpose_kernel`, then set `kStudentKernelImplemented = true`.
