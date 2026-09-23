# Isolated KDA Task: L2/043_mamba_chunk_scan_with_segsum

Run ID: `formal-kda-h100-20260920--sol_execbench--L2-043_mamba_chunk_scan_with_segsum`  
Benchmark: `sol_execbench`

This workspace intentionally exposes only the official definition, five fixed feedback workloads,
the candidate source location, and the trusted evaluation launcher. It contains no prior optimized solution.

```bash
./scripts/evaluate_candidate.sh feedback c001
```

The trusted controller selects and locks an empty GPU, rejects foreign-process interference,
checks immutable candidate IDs and hashes, and invokes the benchmark-specific official evaluator.

Final full evaluation is operator-only:

```bash
./scripts/evaluate_candidate.sh final <candidate-id>
```
