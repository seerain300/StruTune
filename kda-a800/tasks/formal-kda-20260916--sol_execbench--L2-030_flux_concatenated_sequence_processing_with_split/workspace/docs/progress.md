# Progress / Decision Log — L2/030

Append-only human-readable lineage notes. Canonical evidence is `candidates.jsonl`.

## c001 (root) — eval #1

- Config: `tf32x3`, two launches, `BM128/BN128/BK32/warps4/stages3`, `GROUP_M8`.
- Result: **PASS 5/5**, geomean **0.3475×** (all workloads 0.30–0.42×, i.e. ~3× slower than ref).
- Accuracy: `max_abs ≈ 1.4–2.0e-5` on every workload → near-fp32, comfortably inside atol.
- **Assumption checks (draft §5, H1):**
  - Reference is **true fp32**: achieved throughput ≈ 110 GFLOP / 6.08 ms ≈ **18 TFLOPS** on WL1,
    right at the A800 fp32 SIMT peak (~19.5). Confirms the reference is NOT TF32-accelerated, so
    the accuracy assumption holds and there IS headroom for a tensor-core path.
  - Our kernel achieved only ≈ 110 GFLOP / 17.75 ms ≈ **6.2 TFLOPS** — far below the tf32x3 ceiling
    (~52 TFLOPS eff). So the bottleneck is **our tiling/occupancy**, not the algebra or precision.
- **Diagnosis (why so slow):** `BLOCK_K=32` with `num_warps=4` and `tf32x3` (3× K-work) gives a very
  short/inefficient K-loop and low MMA utilization; single-tile-per-CTA output with small blocks
  underuses the 108 SMs poorly for these M sizes. The fix is a proper GEMM tiling, not a redesign.
- **Decision:** `root`, keep as baseline. H1 partially confirmed (correctness + fp32 ref) but c001
  is slower than ref → do NOT abandon; the falsifier (ref already TF32) is ruled out by the 18 TFLOPS
  reference measurement. Proceed to **Phase C tiling sweep** with larger `BLOCK_K` and `num_warps`.

### Next candidate (planned, not yet implemented)
- **c002** — same fused two-launch `tf32x3`, retune tiling toward a standard efficient A800 GEMM:
  `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=3, GROUP_M=8` (change multiple
  interacting occupancy knobs together as one "tiling" axis step, since BK/warps/BN co-determine MMA
  efficiency). Hypothesis H3/H6: raising `BLOCK_K` to 64 and `num_warps` to 8 lifts MMA utilization
  from ~6 TFLOPS toward the tf32x3 ceiling. If smem overflows on `sm_80`, reduce `num_stages` or
  `BLOCK_N` under a new id (record the compile failure as a negative result).

## c002 (parent c001) — eval #2

- Single-axis change from c001: **`num_warps` 4 → 8** (all else identical). Kept it single-axis
  rather than the multi-knob jump sketched above, for clean attribution and zero compile risk.
- Result: **PASS 5/5**, geomean **0.6273×** (was 0.3475× at c001) — **1.80× improvement** from one
  knob. Per-workload 0.52–0.74×. Accuracy identical to c001 (`max_abs` unchanged).
- Confirms c001 was **warp/register-bound**: a 128×128 fp32 accumulator over 4 warps ≈ 128
  acc-regs/thread (spilling); 8 warps halves that and doubles MMA parallelism. **new-best.**
- Still **< 1.0×** vs the fp32 reference → more tiling headroom remains. Likely levers next:
  - `BLOCK_K=32` gives a short K-loop; with `tf32x3` (3× K-work) a larger `BLOCK_K` (64) should
    improve compute/load overlap.
  - `BLOCK_N` could grow (256) to raise arithmetic intensity / W-reuse per tile.

### Next candidate (planned, not yet implemented)
- **c003** — from c002 (new best), single tiling-axis step: **`BLOCK_K` 32 → 64** (keep
  `BM128/BN128/warps8/stages3/GROUP_M8`). H6: longer K-tile improves MMA/load overlap under the
  3× K-cost of `tf32x3`. Watch smem (fp32 tiles): if it fails to compile on `sm_80`, record the
  negative result and instead try `BLOCK_N=256` or `num_stages=4` under a new id.
