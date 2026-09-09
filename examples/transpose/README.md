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

```
transpose of 8192 x 4096 float into 4096 x 8192, 20 timed iterations
implementation       verify     mismatches    time_us     GB/s  vs_cublas
transpose_cublas     PASS                0      752.6    356.7      1.00x
transpose_cutensor   -                 n/a        n/a      n/a        n/a
transpose_cudnn      PASS                0     4031.9     66.6      0.19x
transpose            PASS                0      748.0    358.9      1.01x
```

Shape sweep (GB/s, 20 timed iterations, all verified `PASS`):

| shape (m x n)    | cublas | cudnn | transpose | vs cublas |
|------------------|--------|-------|-----------|-----------|
| 256 x 256        | 37.3   | 29.5  | 32.3      | 0.87x     |
| 2048 x 2048      | 408.6  | 64.5  | 402.1     | 0.98x     |
| 4096 x 4096      | 339.0  | 64.4  | 320.7     | 0.95x     |
| 8192 x 8192      | 362.7  | 64.6  | 370.5     | 1.02x     |
| 16384 x 1024     | 332.9  | 65.5  | 339.7     | 1.02x     |
| 32768 x 512      | 335.1  | 64.3  | 362.0     | 1.08x     |
| 65536 x 128      | 356.1  | 65.8  | 364.8     | 1.02x     |
| 1024 x 16384     | 371.2  | 65.9  | 371.7     | 1.00x     |
| 512 x 32768      | 369.9  | 66.6  | 367.8     | 0.99x     |
| 128 x 65536      | 350.2  | 64.6  | 367.0     | 1.05x     |

cuTENSOR was not installed for these runs (`n/a` throughout).

## Reading the numbers

- **verify** — output compared bit-exactly against the input; a transpose only
  moves floats, so any difference is a wrong address rather than rounding.
- **mismatches** — elements that differ from the bit-exact expectation.
- **time_us** — mean over 20 timed iterations (3 warmup iterations first).
- **GB/s** — prices every implementation at 2 × m × n × 4 bytes per iteration
  (one read of the input, one write of the output), the least a transpose can
  move.
- **vs_cublas** — time normalized to the cuBLAS baseline.

**Takeaway:** at every size that saturates memory bandwidth the CuTe kernel
sits within ±8% of cuBLAS, slightly ahead on extreme aspect ratios (1.08x at
32768 x 512) and slightly behind mid-size squares (0.95x at 4096 x 4096).
At 256 x 256 launch overhead dominates and cuBLAS wins by 13%. cuDNN is ~5x
slower throughout, never passing ~66 GB/s.

## Build & Run

```bash
cmake -B build && cmake --build build
./build/examples/transpose                              # 8192 x 4096, 20 iterations
./build/examples/transpose --m=32768 --n=512            # m, n must be multiples of 32
```
