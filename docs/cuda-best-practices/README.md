# CUDA C++ Best Practices Guide 学习笔记

官方文档：

- 最新版入口: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

## 实验优先入口

- [实验总览](./experiments/README.md)

当前已经落地的核心实验：

| 主题 | 核心问题 | 入口 |
| --- | --- | --- |
| 计时基线 | `std::chrono`、`cudaEvent`、端到端时间分别量到了什么 | [计时基线](./experiments/README.md#计时基线) |
| Pageable vs Pinned | 为什么 pinned host memory 的 `H2D / D2H` 带宽更高 | [Pageable vs Pinned](./experiments/README.md#pageable-vs-pinned) |
| Multi Stream Overlap | copy 和 compute 为什么能 overlap | [Multi Stream Overlap](./experiments/README.md#multi-stream-overlap) |
| Offset vs Stride Copy | coalescing 为什么对 stride 访问特别敏感 | [Offset vs Stride Copy](./experiments/README.md#offset-vs-stride-copy) |
| Shared Memory GEMM | tile 复用和 bank conflict 怎么影响性能 | [Shared Memory GEMM](./experiments/README.md#shared-memory-gemm) |
| L2 Access Window | `accessPolicyWindow` 为什么会提速，也为什么会 thrash | [L2 Access Window](./experiments/README.md#l2-access-window) |
| Async Copy to Shared | `cp.async` 什么时候值得用 | [Async Copy](./experiments/README.md#asynchronous-copy-from-global-memory-to-shared-memory) |
| Occupancy Sweep | `block size / registers / shared memory` 怎么共同影响表现 | [Occupancy Sweep](./experiments/README.md#occupancy-sweep) |
| Instruction Optimization | `half2 / __fdividef / rsqrt / strength reduction` 的收益边界 | [Instruction Optimization](./experiments/README.md#instruction-optimization) |
| Branch / Scheduler | warp divergence、compute-both、ILP、occupancy 的关系 | [Branch / Scheduler](./experiments/README.md#branch--scheduler) |

这份子文档的目标不是翻译官方手册，而是把它整理成一条适合本仓库的学习路线：

1. 先提炼每章真正值得记的结论。
2. 再把能落地的优化点转成可复现的小实验。
3. 最后把实验结果回接到 LeetGPU 题解和后续 CUDA 代码实现。

## 怎么用

- 第一次读，按官方推荐顺序顺着过一遍，不要直接跳到 memory/occupancy。
- 第二次读，只保留“会影响写代码决策”的结论，不抄定义。
- 每读完一个主题，至少补一个最小实验，确认自己真的观察到了现象。
- 每个实验都记录 baseline、改动点、指标、结论，避免只记“好像更快了”。

## 推荐学习顺序

### 第一轮：建立优化框架

1. Preface
2. Application Profiling
3. Getting the Right Answer
4. Optimizing CUDA Applications
5. Performance Metrics

这一轮先建立 APOD 习惯：先找热点，再验证正确性，再度量，再优化。

### 第二轮：进入性能核心

1. Memory Optimizations
2. Execution Configuration Optimizations
3. Instruction Optimization
4. Control Flow

这一轮是后面复现实验的重点，也是最容易直接映射到 LeetGPU 题目的部分。

### 第三轮：补工程化与部署

1. Deploying CUDA Applications
2. Understanding the Programming Environment
3. CUDA Compatibility Developer's Guide
4. Preparing for Deployment
5. Deployment Infrastructure Tools
6. Recommendations and Best Practices
7. nvcc Compiler Switches

这些章节不一定直接提速，但会决定代码怎么编、怎么发、怎么兼容不同机器。

## 章节速记

### 1. Overview / 2.Preface

- 核心方法是 APOD: Assess, Parallelize, Optimize, Deploy。
- Assess：分析哪块代码占了大量耗时。
- Parallelize：并行化，包括CPU代码迁移到GPU，使用NV库等。
- Optimize：
  - 优化是迭代过程，不是一次性把所有技巧堆上去。
  - 计算通信重叠到浮点指令都是优化手段。
- Deploy：见好就收，重新Access。

### 3. Heterogeneous Computing

- CPU 和 GPU 分工不同，适合把并行度高、吞吐导向的部分放到设备端。
  - GPU线程多，A100支持108sm*64warp*32thread=221184线程。
  - GPU没有寄存器切换开销，每个active线程都有单独寄存器。
- 在决定上 GPU 之前，先确认数据搬运成本和热点比例是否值得。
  - 计算除通信如果O(1)就不值，如矩阵相加O(N)/O(N)不值，矩阵相乘O(N^3)/O(N)值。
  - 尽量避免GPU到CPU拷贝，因为PCIE带宽低。

### 4. Application Profiling

- 无论使用什么工具，profile目标都是发现热点/瓶颈代码。
- 要区分强缩放、弱缩放，不同问题规模下结论可能不一样。

### 5. Parallelizing Your Application / 6. Getting Started

- 能用库先用库，比如 cuBLAS、cuFFT、Thrust。
- 真正自己写 kernel 时，要优先暴露并行性，而不是先纠结细枝末节。

### 7. Getting the Right Answer

- \_\_host__ \_\_device__ 函数多写，有利于UT。
- 正确性验证必须和优化并行推进，不要最后才补。
- 浮点计算不满足结合律，CPU/GPU 或不同优化版本结果不完全一致是常态。
- 每个实验都应该保留 reference 实现和误差阈值。

### 8. Optimizing CUDA Applications

- 优化顺序应该从大头开始：访存、并行度、数据传输、执行配置，再到指令级微调。
- 每次优化都要测量收益，不接受“理论上更快”。

### 9. Performance Metrics

- 先统一计时方法，再比较优化效果。
- 建议同时记录 kernel 时间、端到端时间、有效带宽。
- 理论带宽和有效带宽要分开看，前者是上限，后者才是实际结果。

### 10. Memory Optimizations

- Host/Device 传输是第一层瓶颈，优先关注 pinned memory、异步拷贝、overlap。
- Global memory 重点看 coalescing、misalignment、stride。
- Shared memory 重点看 bank conflict、tile 复用、是否值得用 `cp.async`。
- Register pressure 和 occupancy 是联动关系，寄存器不是越多越好。

### 11. Execution Configuration Optimizations

- block size 不是越大越快，要结合 occupancy、访存模式、寄存器压力一起看。
- occupancy 很重要，但不是唯一目标；高 occupancy 不等于高性能。
- shared memory 和并发 kernel 会直接影响调度空间。

### 12. Instruction Optimization

- 指令级优化要放在后面做，只有前面的瓶颈已经收敛时才值得深挖。
- 常见点包括除法/取模替换、rsqrt、fast math、循环计数器类型。

### 13. Control Flow

- warp divergence 是典型吞吐杀手，尤其在分支高度不均匀时。
- predication 能缓解部分短分支，但不是万能解法。

### 14-20. 部署、环境、兼容性、工具链

- 这部分更偏工程实践：编译目标、兼容性策略、错误处理、分发方式。
- 如果后面要把实验搬到不同显卡或不同 CUDA 版本机器上，这部分必须补。

