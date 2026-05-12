# CUDA Practice Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `practice/` CUDA exercise workspace that lets a learner immediately edit kernels, compile, run correctness checks, and see timing.

**Architecture:** The original guidebook remains read-only. New practice code lives under `practice/`, with shared headers in `practice/common/` and standalone executable exercises in numbered directories.

**Tech Stack:** CMake, CUDA C++17, CUDA Runtime API, `cudaEvent` timing.

---

### Task 1: Build System And Shared Helpers

**Files:**
- Create: `practice/CMakeLists.txt`
- Create: `practice/common/cuda_check.h`
- Create: `practice/common/timer.h`
- Create: `practice/common/validation.h`

- [x] **Step 1: Add CMake project**

Create a CUDA CMake project with one executable target per practice directory and shared include paths.

- [x] **Step 2: Add CUDA error checking**

Create `CUDA_CHECK(expr)` so runtime failures include file, line, CUDA error name, and message.

- [x] **Step 3: Add GPU timer**

Create a small RAII wrapper around CUDA events for kernel timing.

- [x] **Step 4: Add validation helpers**

Create float comparison and mismatch reporting helpers used by exercises.

### Task 2: Documentation And Sanity Check

**Files:**
- Create: `practice/README.md`
- Create: `practice/00-sanity-check/README.md`
- Create: `practice/00-sanity-check/cuda_sanity.cu`

- [x] **Step 1: Write practice README**

Document configure, build, run, and learning order commands.

- [x] **Step 2: Add solved CUDA sanity executable**

Add a minimal kernel that writes indices to device memory and verifies the result on host.

### Task 3: Starter Exercises

**Files:**
- Create: `practice/01-vector-add/README.md`
- Create: `practice/01-vector-add/exercise.cu`
- Create: `practice/02-matrix-copy/README.md`
- Create: `practice/02-matrix-copy/exercise.cu`
- Create: `practice/03-matrix-transpose/README.md`
- Create: `practice/03-matrix-transpose/exercise.cu`
- Create: `practice/04-reduction/README.md`
- Create: `practice/04-reduction/exercise.cu`
- Create: `practice/05-softmax/README.md`
- Create: `practice/05-softmax/exercise.cu`

- [x] **Step 1: Add TODO kernels**

Each `exercise.cu` contains one clearly marked kernel for the learner to implement.

- [x] **Step 2: Add CPU references**

Each exercise computes expected output on the CPU.

- [x] **Step 3: Add correctness checks and timing**

Each exercise prints a TODO message until `kStudentKernelImplemented` is set to `true`; after that it runs the kernel, checks results, and prints kernel time.

### Task 4: Verification

**Files:**
- Verify all created `practice/` files.

- [x] **Step 1: Configure**

Run: `cmake -S practice -B practice/build`

- [x] **Step 2: Build**

Run: `cmake --build practice/build -j`

- [x] **Step 3: Run sanity check**

Run: `./practice/build/00-sanity-check/cuda_sanity`

- [x] **Step 4: Run one starter exercise**

Run: `./practice/build/01-vector-add/vector_add`

Expected: exits with TODO guidance because the learner has not implemented the kernel yet.
