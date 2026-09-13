# Baseline checker examples

The preliminary checker matrix combines token counts 7, 10, and 17; world sizes 1, 2, 3, and 4; and contiguous or round-robin sharding. Implementations include a reference, main-process-only consumption, and local-size trimming. A fixed master seed is used, with changed-seed and changed-token controls.

Measurements include token multisets, master-seed/token-derived streams, serialized payload lengths, per-token outcomes and scores, and aggregate means. The reference preserves the contract across all 24 configurations. Faulty forms expose discrepancies in nondivisible configurations. Changes to the scientific input are distinguished from execution-topology changes. The larger 48-mutant analysis is reported separately.
