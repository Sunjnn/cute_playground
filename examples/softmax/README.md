# softmax

Row-wise softmax, five implementations benchmarked head-to-head on an
**NVIDIA RTX 5060** (Blackwell, sm_120). The two CuTe kernels (`softmax` and
`softmax_multistage`) are implemented in this repo; the other three are
third-party baselines they are measured against.

## Baseline vs. this implementation

Both CuTe kernels are implemented in this repo; the benchmark pitches the
multistage one against the single-buffer one and against third-party
libraries.

- **`softmax_multistage` — this repo's optimized implementation.** Same tiling
  and compute as `softmax`, but the shared-memory tile buffer is a **4-stage
  ring**. A prologue prefetches the first tiles with `cp.async`, the steady
  state issues the load for tile *i + 4* while tile *i* is being computed,
  and an epilogue drains the remaining stages — global loads and compute
  overlap instead of serializing.
- **`softmax` — this repo's baseline.** Single shared-memory buffer: each
  1×2048 tile is copied in with `cp.async`, fully waited on, computed, then
  copied out before the next tile's load begins. The `vs_softmax` column is
  normalized against this kernel.
- `softmax_cub` — external baseline: CUB `DeviceSegmentedReduce` + thrust
  transforms.
- `softmax_fmha` — external baseline: the CUTLASS FMHA-style two-pass kernel
  (as in `fmha_collective_softmax.hpp`).
- `softmax_cudnn` — external baseline: cuDNN `cudnnSoftmaxForward` (optional
  dependency; without cuDNN it reports `n/a`, see the root README).

## Results

```
softmax of 8192 x 8192 float, 20 timed iterations
implementation       verify    max_rel_diff    time_us     GB/s vs_softmax
softmax              PASS         3.054e-07     1653.6    487.0      1.00x
softmax_multistage   PASS         3.105e-07     1415.0    569.1      1.17x
softmax_cub          PASS         4.015e-07     2071.2    388.8      0.80x
softmax_fmha         PASS         4.459e-07     1702.6    473.0      0.97x
softmax_cudnn        PASS         3.113e-07     1409.1    571.5      1.17x
```

## Reading the numbers

- **verify** — output compared against a double-precision row-wise reference;
  `PASS` means `max_rel_diff ≤ 1e-4`.
- **max_rel_diff** — largest relative deviation from that reference.
- **time_us** — mean over 20 timed iterations (3 warmup iterations first).
- **GB/s** — prices every implementation at 3 × M × N × 4 bytes per
  iteration (two reads of the input, one write of the output), the least a
  two-pass softmax can cost.
- **vs_softmax** — time normalized to the single-buffer `softmax` baseline.

**Takeaway:** overlapping loads with compute through the 4-stage pipeline
lifts the baseline from 487.0 to 569.1 GB/s (**1.17×**), within ~0.4% of
cuDNN and well ahead of CUB and the FMHA-style kernel.

## Build & Run

```bash
cmake -B build && cmake --build build
./build/examples/softmax                                  # 8192 x 8192, 20 iterations
```
