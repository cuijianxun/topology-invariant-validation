# Topology-Invariant Validation Contracts

Reproduction materials for **Topology-Invariant Validation Contracts for Distributed Driving-World-Model Evaluation**.

Authors: Jianxun Cui, Jinlong Cui, Marko Milojkovic, Stanisa Peric, and Vladan Devedzic.

The checker tests whether a change in evaluation topology preserves scenario tokens, random-stream assignments, failure accounting, and metric reduction. This repository contains the implementation, controlled benchmarks, public-source provenance, recorded results, and figure sources. It does not include model weights or restricted datasets.

## Quick start

Use Python 3.12. The standard-library tests do not need PyTorch:

```sh
python -B -m unittest discover -s tests -p "test_*.py" -v
```

The native collective experiment additionally requires a CPU build of PyTorch with Gloo support. See `environment.txt` for the recorded environment. PyTorch is an external dependency, not bundled here.

## Reproduce the analyses

Run commands from the repository root. They regenerate files below `results/`; preserve a copy of recorded results if exact comparison is required.

```sh
python -B scripts/topology_checker.py
python -B scripts/mutation_benchmark.py
python -B scripts/evaluation_analysis.py
python -B scripts/historical_defects.py
python -B scripts/extracted_adapters.py
python -B scripts/native_multiprocess.py
python -B scripts/scalability.py
```

For actual distributed collectives, after installing a suitable CPU PyTorch distribution:

```sh
python -B scripts/gloo_topology.py
```

The recorded native Gloo study contains 12 topology cells, with 10 repetitions per cell. Process adapters preserve selected orchestration mechanisms using token fixtures; they do not run the original world models. Timing measurements depend on the host and should not be expected to match recorded milliseconds exactly.

## Recorded results

- Controlled mutation benchmark: 48/48 detections and localizations; strongest generic baseline: 31/48.
- Allowed variations: 0 alarms in 24 cases; Wilson 95% upper bound: 13.80%.
- Historical challenge: all five pre-fix fixtures detected; four source-fixed fixtures accepted. The fifth removes repeated validation samples by discarding the tail, conflicting with the declared full-set population.
- Repository study: 13 eligible cases from 16 candidates; seven contain static topology-sensitive signals. These are source observations, not measurements of original model execution.

## Directory map

| Directory/file | Contents |
|---|---|
| `scripts/`, `tests/` | Checker, analysis, execution harnesses, unit tests |
| `data/` | Revision-pinned repository and historical-defect provenance |
| `experiments/` | Study definitions and interpretation of protocol outcomes |
| `results/` | Recorded machine-readable outputs |
| `paper/figures/` | Vector figures and TikZ/PGFPlots sources |
| `output_map.csv` | Manuscript-to-evidence mapping |
| `CITATION.cff` | Citation metadata; no publication DOI is claimed |
| `MANIFEST.csv`, `CHECKSUMS_SHA256.txt` | File inventory and integrity checks |

Figure sources can be compiled with a LaTeX installation providing TikZ and PGFPlots. The manuscript text and editorial correspondence are not included in this code repository.

## Rights and third-party dependencies

See `RIGHTS_AND_THIRD_PARTY_NOTICES.md`. This release does not assign a new open-source license. Public visibility alone does not grant unrestricted redistribution or reuse rights.
