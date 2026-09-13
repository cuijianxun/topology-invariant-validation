"""Native torch.distributed/Gloo topology-invariance experiment for topology."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as tmp


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "topology_checker.py")
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, repeats: int, output_path: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(minutes=4),
    )
    cases = []
    try:
        for token_count in (17, 23):
            tokens = [f"gloo_{index:03d}_{'q' * (index % 5)}" for index in range(token_count)]
            baseline = GATE.signature(GATE.correct_evaluate(tokens, 1, "contiguous", GATE.MASTER_SEED))
            for sharding in GATE.SHARDINGS:
                parts = GATE.shard(tokens, world_size, sharding)
                durations = []
                relation_passes = []
                for _ in range(repeats):
                    dist.barrier()
                    started = time.perf_counter()
                    local_rows = [GATE.row(token, GATE.MASTER_SEED) for token in parts[rank]]
                    payload = json.dumps(local_rows, sort_keys=True).encode("utf-8")
                    local_size = torch.tensor([len(payload)], dtype=torch.int64)
                    gathered_sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
                    dist.all_gather(gathered_sizes, local_size)
                    max_size = max(int(size.item()) for size in gathered_sizes)
                    local_tensor = torch.zeros(max_size, dtype=torch.uint8)
                    if payload:
                        local_tensor[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
                    gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
                    dist.all_gather(gathered_tensors, local_tensor)
                    dist.barrier()
                    elapsed = time.perf_counter() - started
                    if rank == 0:
                        gathered = [
                            json.loads(bytes(tensor[: int(size.item())].tolist()).decode("utf-8"))
                            for tensor, size in zip(gathered_tensors, gathered_sizes)
                        ]
                        rows = [row for shard_rows in gathered for row in shard_rows]
                        relation_passes.append(GATE.signature(rows) == baseline)
                        durations.append(elapsed)
                if rank == 0:
                    cases.append(
                        {
                            "world_size": world_size,
                            "token_count": token_count,
                            "sharding": sharding,
                            "repeats": repeats,
                            "all_relations_pass": all(relation_passes),
                            "median_seconds": statistics.median(durations),
                            "p95_seconds": sorted(durations)[max(0, int(0.95 * len(durations)) - 1)],
                        }
                    )
        if rank == 0:
            Path(output_path).write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def run_gloo(repeats: int = 10) -> dict:
    if not dist.is_available() or not dist.is_gloo_available():
        return {
            "schema_version": 1,
            "evidence_level": "NATIVE_TORCH_DISTRIBUTED_GLOO",
            "overall_pass": False,
            "gates": {"gloo_available": False},
            "torch_version": torch.__version__,
            "cases": [],
        }

    all_cases = []
    shard_dir = ROOT / "results" / "evaluation" / "gloo_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for world_size in (2, 3, 4):
        output = shard_dir / f"world_size_{world_size}.json"
        tmp.spawn(
            _worker,
            args=(world_size, _free_port(), repeats, str(output)),
            nprocs=world_size,
            join=True,
        )
        all_cases.extend(json.loads(output.read_text(encoding="utf-8")))

    expected_cases = 3 * 2 * len(GATE.SHARDINGS)
    gates = {
        "gloo_available": True,
        "actual_world_sizes_2_3_4": {case["world_size"] for case in all_cases} == {2, 3, 4},
        "complete_topology_matrix": len(all_cases) == expected_cases,
        "ten_repeats_each": repeats >= 10 and all(case["repeats"] >= 10 for case in all_cases),
        "all_four_relations_pass": all(case["all_relations_pass"] for case in all_cases),
    }
    return {
        "schema_version": 1,
        "evidence_level": "NATIVE_TORCH_DISTRIBUTED_GLOO",
        "torch_version": torch.__version__,
        "backend": "gloo",
        "cases": all_cases,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def main() -> None:
    summary = run_gloo()
    output = ROOT / "results" / "evaluation" / "gloo_topology_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    tmp.freeze_support()
    main()
