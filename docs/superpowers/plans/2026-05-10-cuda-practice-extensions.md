# CUDA Practice Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exercises `06-rmsnorm` through `10-tiled-transpose` to the CUDA practice workspace.

**Architecture:** Each exercise is a standalone CUDA executable under `practice/`, matching the existing starter format. Shared validation and timing continue to come from `practice/common/`.

**Tech Stack:** CMake, CUDA C++17, CUDA Runtime API, `cudaEvent` timing.

---

### Task 1: Register New Practice Targets

**Files:**
- Modify: `practice/CMakeLists.txt`
- Modify: `practice/README.md`

- [ ] **Step 1: Add CMake targets**

Add one `add_practice_target(...)` call for each new numbered exercise.

- [ ] **Step 2: Update practice README learning order**

Add entries for `06-rmsnorm` through `10-tiled-transpose`.

### Task 2: Add Transformer Operator Exercises

**Files:**
- Create: `practice/06-rmsnorm/README.md`
- Create: `practice/06-rmsnorm/exercise.cu`
- Create: `practice/07-gelu/README.md`
- Create: `practice/07-gelu/exercise.cu`
- Create: `practice/08-rope/README.md`
- Create: `practice/08-rope/exercise.cu`

- [ ] **Step 1: Add RMSNorm exercise**

Create a row-wise RMSNorm starter with CPU reference and timing.

- [ ] **Step 2: Add GELU exercise**

Create an elementwise GELU starter with CPU reference and timing.

- [ ] **Step 3: Add RoPE exercise**

Create a rotary positional embedding starter with CPU reference and timing.

### Task 3: Add Optimization Exercises

**Files:**
- Create: `practice/09-optimized-reduction/README.md`
- Create: `practice/09-optimized-reduction/exercise.cu`
- Create: `practice/10-tiled-transpose/README.md`
- Create: `practice/10-tiled-transpose/exercise.cu`

- [ ] **Step 1: Add optimized reduction exercise**

Create a starter where each thread should accumulate multiple elements before reducing within the block.

- [ ] **Step 2: Add tiled transpose exercise**

Create a shared-memory tiled transpose starter with a padded tile declaration.

### Task 4: Verification

**Files:**
- Verify all files under `practice/06-rmsnorm` through `practice/10-tiled-transpose`.

- [ ] **Step 1: Configure**

Run: `cmake -S practice -B practice/build`

- [ ] **Step 2: Build**

Run: `cmake --build practice/build -j`

- [ ] **Step 3: Run new exercises**

Run each new executable and verify it prints TODO guidance:

```bash
./practice/build/06-rmsnorm/rmsnorm
./practice/build/07-gelu/gelu
./practice/build/08-rope/rope
./practice/build/09-optimized-reduction/optimized_reduction
./practice/build/10-tiled-transpose/tiled_transpose
```
