# 15 - Embedding Lookup

Implement embedding lookup:

```cpp
output[token, dim] = table[token_ids[token], dim]
```

The embedding table has shape `vocab_size x embedding_dim`.

## What To Practice

- Gather-style memory access.
- Mapping a linear thread index to `(token, dim)`.
- Understanding non-contiguous reads and contiguous writes.

## Run

```bash
cmake --build practice/build -j
./practice/build/15-embedding-lookup/embedding_lookup
```

Edit `exercise.cu`, implement `embedding_lookup_kernel`, then set `kStudentKernelImplemented = true`.
