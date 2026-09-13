# Mutation and allowed-variation definitions

`scripts/mutation_benchmark.py` defines the executable cases used by `scripts/evaluation_analysis.py`.

- Scenario multiset: 12 cases involving drops, duplication, replacement, renaming, truncation, subset selection, and empty results.
- Random stream: 12 cases involving rotation, reversal, shared streams, rank-local or rank-derived streams, swapped or truncated keys, and position binding.
- Failure accounting: 12 cases involving dropped failures, cleared or added failures, swapped outcomes, missing fields, and invalid status values.
- Aggregate reduction: 12 cases involving stale values, failures treated as zero, rank-only subsets, unweighted shard means, removed maxima, rounding, alternative reducers, score reassignment, and out-of-tolerance differences.

The 24 allowed variations include two row permutations, six diagnostic-metadata changes, eight rank/timing changes, and eight within-tolerance score perturbations. Absolute numerical tolerance is `1e-9`.
