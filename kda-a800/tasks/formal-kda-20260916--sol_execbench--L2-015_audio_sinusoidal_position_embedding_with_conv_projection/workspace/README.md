# Isolated KDA Task: L2/015_audio_sinusoidal_position_embedding_with_conv_projection

Run ID: `formal-kda-20260916--sol_execbench--L2-015_audio_sinusoidal_position_embedding_with_conv_projection`  
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
