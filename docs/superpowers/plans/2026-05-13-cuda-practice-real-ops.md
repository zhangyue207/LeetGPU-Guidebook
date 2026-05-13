# CUDA Practice Real Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add practical CUDA operator exercises `11-saxpy` through `15-embedding-lookup`.

**Architecture:** Each exercise is a standalone CUDA executable under `practice/`, matching the existing starter format. Shared CUDA error checking, timing, and float validation remain in `practice/common/`.

**Tech Stack:** CMake, CUDA C++17, CUDA Runtime API, `cudaEvent` timing.

---

### Task 1: Register Exercises

**Files:**
- Modify: `practice/CMakeLists.txt`
- Modify: `practice/README.md`

- [ ] **Step 1: Add CMake targets**

Add targets for `11-saxpy`, `12-layernorm`, `13-vector-dot`, `14-histogram`, and `15-embedding-lookup`.

- [ ] **Step 2: Update learning order**

Add the five exercises to the practice README with one-line learning goals.

### Task 2: Add Operator Starters

**Files:**
- Create: `practice/11-saxpy/README.md`
- Create: `practice/11-saxpy/exercise.cu`
- Create: `practice/12-layernorm/README.md`
- Create: `practice/12-layernorm/exercise.cu`
- Create: `practice/13-vector-dot/README.md`
- Create: `practice/13-vector-dot/exercise.cu`
- Create: `practice/14-histogram/README.md`
- Create: `practice/14-histogram/exercise.cu`
- Create: `practice/15-embedding-lookup/README.md`
- Create: `practice/15-embedding-lookup/exercise.cu`

- [ ] **Step 1: Add SAXPY**

Create an in-place `y[i] = alpha * x[i] + y[i]` starter.

- [ ] **Step 2: Add LayerNorm**

Create a row-wise LayerNorm starter with CPU reference.

- [ ] **Step 3: Add vector dot**

Create a dot-product starter with block partial sums.

- [ ] **Step 4: Add histogram**

Create a histogram starter using integer output bins.

- [ ] **Step 5: Add embedding lookup**

Create a gather-style embedding lookup starter.

### Task 3: Verify

**Files:**
- Verify all new practice files.

- [ ] **Step 1: Build**

Run: `cmake --build practice/build -j`

- [ ] **Step 2: Run new exercises**

Each new executable should print TODO guidance and exit with code `2`.
