# CUDA Practice Workspace Design

## Goal

Add a beginner-friendly `practice/` workspace to this repository so CUDA kernels can be written, built, run, checked, and timed without modifying the original guidebook notes or solution files.

## Shape

The workspace keeps original repository content intact and adds:

- `practice/CMakeLists.txt` for one-command builds.
- `practice/common/` for CUDA error checking, event timing, and float comparison helpers.
- `practice/00-sanity-check/` as a solved environment check.
- `practice/01-vector-add/` through `practice/05-softmax/` as kernel-writing exercises.

Each exercise is a standalone executable. The student edits only the local `exercise.cu` kernel and flips `kStudentKernelImplemented` to `true` when ready to run correctness checks.

## Learning Flow

The exercises progress through the minimum CUDA kernel-writing path:

1. Vector add: one-dimensional indexing and bounds checks.
2. Matrix copy: two-dimensional grid/block indexing.
3. Matrix transpose: row/column mapping and memory access patterns.
4. Reduction: block-local reduction and partial sums.
5. Softmax: stable max/sum reduction inside each row.

## Verification

The workspace must support:

- Configure and build with CMake.
- Run `00-sanity-check` successfully to validate CUDA runtime availability.
- Run each exercise executable and get a clear TODO message until the student implements the kernel.
- After implementation, each exercise compares GPU output against a CPU reference and prints kernel timing.

