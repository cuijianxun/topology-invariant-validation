# Evaluation protocol

The primary comparison uses the same 48 single-family mutants for the four-identity contract, schema validation, scalar golden checking, and row-count-plus-scalar equality. The primary outcome is paired fault detection; secondary outcomes are diagnostic accuracy and alarms on allowed variations. Exact McNemar testing compares paired detections. Wilson intervals summarize binomial proportions.

The design fixes four fault families and 24 allowed variations. The original numerical targets were contract detection of at least 90%, an improvement of at least 15 percentage points over the strongest baseline, and an observed allowed-variation alarm fraction no greater than 5%. These are design criteria, not population guarantees. The observed 0/24 alarms has a Wilson 95% upper bound of 13.80%.

Repository, historical, native-backend, adapter, and scalability evidence remain separate. Native Gloo uses world sizes 2, 3, and 4, two token counts, two sharding schemes, and ten repetitions per cell. Three extracted process adapters use token fixtures in place of original models and datasets. Scalability uses 100, 1,000, 10,000, and 100,000 tokens, logical world sizes 1, 2, 4, and 8, and 30 repetitions per cell.

The recorded implementation and benchmark definitions specify the individual cases. No result or unfavorable case was removed in preparing this release.
