# CUDA Practice Parallel Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CUDA practice exercises `16-prefix-sum` through `20-max-pooling`.

**Architecture:** Each new exercise is a standalone executable under `practice/`, reusing shared CUDA check, timing, and validation helpers. Exercises start as TODO kernels and validate against CPU references after the learner flips the implementation boolean.

**Tech Stack:** CMake, CUDA C++17, CUDA Runtime API, `cudaEvent` timing.

---

### Task 1: Register Exercises

**Files:**
- Modify: `practice/CMakeLists.txt`
- Modify: `practice/README.md`

- [ ] **Step 1: Add CMake targets**

Add `add_practice_target(...)` entries for `16-prefix-sum`, `17-matrix-vector`, `18-tiled-matmul`, `19-conv2d`, and `20-max-pooling`.

- [ ] **Step 2: Update learning order**

Add exercise descriptions for `16` through `20`.

### Task 2: Add Parallel Operator Starters

**Files:**
- Create: `practice/16-prefix-sum/README.md`
- Create: `practice/16-prefix-sum/exercise.cu`
- Create: `practice/17-matrix-vector/README.md`
- Create: `practice/17-matrix-vector/exercise.cu`
- Create: `practice/18-tiled-matmul/README.md`
- Create: `practice/18-tiled-matmul/exercise.cu`
- Create: `practice/19-conv2d/README.md`
- Create: `practice/19-conv2d/exercise.cu`
- Create: `practice/20-max-pooling/README.md`
- Create: `practice/20-max-pooling/exercise.cu`

- [ ] **Step 1: Add prefix sum**

Create a one-block inclusive scan starter.

- [ ] **Step 2: Add matrix-vector multiplication**

Create a row-wise reduction starter.

- [ ] **Step 3: Add tiled matrix multiplication**

Create a shared-memory tiled GEMM starter.

- [ ] **Step 4: Add 2D convolution**

Create a single-channel `3x3` convolution starter.

- [ ] **Step 5: Add max pooling**

Create a `2x2` stride-2 max-pooling starter.

### Task 3: Verify

**Files:**
- Verify all new practice files.

- [ ] **Step 1: Build**

Run: `cmake --build practice/build -j`

- [ ] **Step 2: Run new exercises**

Each new executable should print TODO guidance and exit with code `2`.
