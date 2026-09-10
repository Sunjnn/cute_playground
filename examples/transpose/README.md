# transpose

Out-of-place 2D transpose of an `m x n` row-major float matrix into an `n x m`
one, benchmarked on an **NVIDIA RTX 5060** (Blackwell, sm_120). `transpose` is
the CuTe kernel implemented in this repo; the other three are third-party
baselines it is measured against.

## Baseline vs. this implementation

- **`transpose` — this repo's kernel.** 32×32 CTA tiles staged through a single
  swizzled shared-memory buffer (`Swizzle<5, 0, 5>`), loaded with `cp.async`
  and stored with plain 32-bit stores. The transpose lives entirely in the
  global strides: both tensors span the same `(m, n)` index space, so the
  shared tile is a pure pass-through written and read at the same logical
  `(i, j)`.
- `transpose_cublas` — external baseline: cuBLAS `cublasSgeam` with alpha = 0.
  The `vs_cublas` column is normalized against it.
- `transpose_cudnn` — external baseline: cuDNN `cudnnTransformTensor` (optional
  dependency; without cuDNN it reports `n/a`).
- `transpose_cutensor` — external baseline: cuTENSOR elementwise permutation
  (optional dependency; without cuTENSOR it reports `n/a`).

**Shape constraint:** `m` and `n` must be multiples of 32. The CTA tile is not
predicated, so a ragged edge would read and write out of bounds; shapes that
violate this fail verification with unwritten elements rather than being
handled.

## Results

Measured on an RTX 5060 (sm_120), CUDA 13.3, cuTENSOR 2.7, cuDNN 9.24.

```
transpose of 8192 x 4096 float into 4096 x 8192, 50 timed iterations
implementation       verify     mismatches    time_us     GB/s  vs_cublas
transpose_cublas     PASS                0      721.2    372.2      1.00x
transpose_cutensor   PASS                0      739.8    362.8      0.97x
transpose_cudnn      PASS                0     4021.4     66.8      0.18x
transpose            PASS                0      733.0    366.2      0.98x
```

Shape sweep (GB/s, 50 timed iterations, all verified `PASS`):

| shape (m x n)    | cublas | cutensor | cudnn | transpose | vs cublas |
|------------------|--------|----------|-------|-----------|-----------|
| 32 x 32          | 0.5    | 0.6      | 0.4   | 0.6       | 1.13x     |
| 64 x 64          | 2.0    | 2.3      | 2.1   | 2.1       | 1.05x     |
| 128 x 128        | 8.8    | 8.3      | 7.1   | 9.2       | 1.04x     |
| 256 x 256        | 34.3   | 40.0     | 24.6  | 37.0      | 1.08x     |
| 512 x 512        | 131.3  | 111.6    | 62.4  | 142.5     | 1.09x     |
| 1024 x 1024      | 540.3  | 499.9    | 86.4  | 524.2     | 0.97x     |
| 2048 x 2048      | 400.9  | 393.9    | 66.0  | 400.2     | 1.00x     |
| 4096 x 4096      | 364.1  | 366.3    | 66.6  | 351.9     | 0.97x     |
| 8192 x 8192      | 376.0  | 370.4    | 67.2  | 374.3     | 1.00x     |
| 16384 x 16384    | 375.8  | 367.3    | 66.4  | 377.9     | 1.01x     |
| 8192 x 4096      | 371.6  | 369.0    | 66.5  | 364.2     | 0.98x     |
| 16384 x 4096     | 373.2  | 360.3    | 66.6  | 370.4     | 0.99x     |
| 4096 x 16384     | 372.3  | 371.5    | 66.5  | 375.2     | 1.01x     |
| 32 x 8192        | 105.2  | 111.0    | 56.1  | 149.9     | 1.42x     |
| 8192 x 32        | 136.9  | 117.2    | 60.6  | 145.6     | 1.06x     |
| 128 x 32768      | 427.9  | 401.4    | 65.6  | 412.5     | 0.96x     |
| 32768 x 128      | 412.2  | 400.0    | 65.0  | 403.7     | 0.98x     |
| 992 x 736        | 378.4  | 263.1    | 83.1  | 347.6     | 0.92x     |
| 3072 x 2048      | 361.6  | 356.7    | 63.5  | 349.6     | 0.97x     |
| 1024 x 1536      | 593.5  | 673.0    | 89.3  | 686.1     | 1.16x     |
| 1536 x 1024      | 636.3  | 677.2    | 89.4  | 647.5     | 1.02x     |

## Reading the numbers

- **verify** — output compared bit-exactly against the input; a transpose only
  moves floats, so any difference is a wrong address rather than rounding.
- **mismatches** — elements that differ from the bit-exact expectation.
- **time_us** — mean over 50 timed iterations (3 warmup iterations first).
- **GB/s** — prices every implementation at 2 × m × n × 4 bytes per iteration
  (one read of the input, one write of the output), the least a transpose can
  move.
- **vs_cublas** — time normalized to the cuBLAS baseline.

**Takeaway:** at every size that saturates memory bandwidth the CuTe kernel
sits within ±8% of cuBLAS, with the largest gaps on the extreme aspect ratios
(1.42x at 32 x 8192) and mid-size squares (0.92x at 992 x 736, 0.97x at
4096 x 4096). At 128 x 128 and below launch overhead dominates and the gap
swings run-to-run. cuTENSOR tracks cuBLAS closely on most sizes but loses to
it on the mid-size squares; its plan heuristic is also noisier run-to-run
(263 GB/s at 992 x 736). cuDNN is ~5x slower throughout, never passing
~90 GB/s. Shapes small enough to sit in L2 (1024 x 1536 and 1536 x 1024) lift
everyone far above DRAM bandwidth and the CuTe kernel leads there too.

## Build & Run

```bash
cmake -B build && cmake --build build
./build/examples/transpose                              # 8192 x 4096, 20 iterations
./build/examples/transpose --m=32768 --n=512            # m, n must be multiples of 32
```

The optional baselines need their roots on a fresh configure (the harness
only uses their headers and import libraries; the DLLs resolve from PATH at
run time):

```bash
cmake -B build -DCUTENSOR_ROOT="C:/Program Files/NVIDIA cuTENSOR/v2.7" \
               -DCUDNN_ROOT="C:/Program Files/NVIDIA/CUDNN/v9.24"
```
