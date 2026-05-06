"""Top-level paper-style experiment runner.

This entry point keeps the paper's experiment logic: compare a rule-based
baseline, an expert controller, direct SAC, and GAIL/GASAC on the same RC
residential AC + PV-battery environment. The implementation uses the current
repository's environment-consistent heuristic expert by default, so the goal is
algorithm validation rather than reproducing the paper's exact numeric results.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gail_control.config import get_config
from gail_control.envs.residential import CustomEnv
from gail_control.imitation.expert_data import heuristic_action
from scripts.compare_control import (
    configure_common,
    cuda_compute_available,
    energy_diagnostics,
    run_baseline,
    run_gasac,
    run_sac,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-style AC and battery control experiment")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device. Falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--seed", type=int, default=9743)
    parser.add_argument("--prices", nargs="+", default=["normal"], choices=["fix", "normal", "dynamic"])
    parser.add_argument("--all_prices", action="store_true", help="Run fix, normal, and dynamic pricing scenarios.")
    parser.add_argument("--quick", action="store_true", help="Use a short validation budget instead of the paper-style budget.")
    parser.add_argument("--gasac_steps", type=int, default=None)
    parser.add_argument("--sac_steps", type=int, default=None)
    parser.add_argument("--demo_min_timesteps", type=int, default=None)
    parser.add_argument("--demo_batch_size", type=int, default=256)
    parser.add_argument("--expert_source", type=str, default="heuristic", choices=["heuristic", "csv", "policy"])
    parser.add_argument("--bc_pretrain_epochs", type=int, default=20)
    parser.add_argument("--bc_posttrain_epochs", type=int, default=10)
    parser.add_argument("--bc_charge_loss_weight", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str, default="results/run_exp")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not cuda_compute_available():
        print("CUDA requested but unavailable for compute; falling back to CPU.")
        return "cpu"
    return requested_device


def budget(args: argparse.Namespace) -> tuple[int, int, int]:
    gasac_steps = args.gasac_steps
    sac_steps = args.sac_steps
    demo_min_timesteps = args.demo_min_timesteps

    if args.quick:
        gasac_steps = 5_000 if gasac_steps is None else gasac_steps
        sac_steps = 15_000 if sac_steps is None else sac_steps
        demo_min_timesteps = 4_608 if demo_min_timesteps is None else demo_min_timesteps
    else:
        gasac_steps = 150_000 if gasac_steps is None else gasac_steps
        sac_steps = 300_000 if sac_steps is None else sac_steps
        demo_min_timesteps = 46_080 if demo_min_timesteps is None else demo_min_timesteps

    return gasac_steps, sac_steps, demo_min_timesteps


def run_expert(config: dict) -> dict:
    """Evaluate the environment-consistent expert controller as M2."""
    start = time.perf_counter()
    env = CustomEnv(config=config, setting="m2_expert_heuristic", flag="test")
    obs, _ = env.reset()
    done = False
    records = []

    while not done:
        action = heuristic_action(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        records.append(info["output"])

    result = pd.DataFrame(records)
    _, metrics = env.evaluation(result, output=True, plot=False)
    elapsed = time.perf_counter() - start
    return {
        "model": "M2_expert_heuristic",
        "training_seconds": 0.0,
        "total_seconds": elapsed,
        **metrics,
        **energy_diagnostics(result),
    }


def add_baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary.loc[summary["model"] == "M1_baseline"].iloc[0]
    summary = summary.copy()
    for idx, row in summary.iterrows():
        summary.loc[idx, "sr_improve"] = (row["sr"] - baseline["sr"]) / baseline["sr"] * 100
        summary.loc[idx, "cost_improve"] = (baseline["cost"] - row["cost"]) / baseline["cost"] * 100
        summary.loc[idx, "sr_vs_baseline_pct"] = summary.loc[idx, "sr_improve"]
        summary.loc[idx, "cost_vs_baseline_pct"] = summary.loc[idx, "cost_improve"]
    return summary


def run_one_price(args: argparse.Namespace, price: str, device: str, output_dir: Path) -> pd.DataFrame:
    gasac_steps, sac_steps, demo_min_timesteps = budget(args)
    base_config = configure_common(get_config([]), device=device, seed=args.seed, price=price)

    results = []
    print(f"\n=== Pricing scenario: {price} ===")

    print("Running M1 PID/rule baseline...")
    baseline_result = run_baseline(copy.deepcopy(base_config))
    results.append(baseline_result)

    base_config["rl_config"].baseline_sr_reference = baseline_result["sr"]
    base_config["rl_config"].baseline_cost_reference = baseline_result["cost"]

    print("Running M2 expert controller...")
    results.append(run_expert(copy.deepcopy(base_config)))

    print(f"Running M4 BC-regularized GASAC/GAIL for {gasac_steps} timesteps...")
    results.append(
        run_gasac(
            copy.deepcopy(base_config),
            gasac_steps,
            args.demo_batch_size,
            demo_min_timesteps,
            args.expert_source,
            args.bc_pretrain_epochs,
            args.bc_posttrain_epochs,
            args.bc_charge_loss_weight,
        )
    )

    print(f"Running M3 direct SAC for {sac_steps} timesteps...")
    results.append(run_sac(copy.deepcopy(base_config), sac_steps))

    summary = add_baseline_comparison(pd.DataFrame(results))
    summary.insert(0, "price_scenario", price)

    scenario_dir = output_dir / price
    scenario_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(scenario_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    device = resolve_device(args.device)
    prices = ["fix", "normal", "dynamic"] if args.all_prices else args.prices
    gasac_steps, sac_steps, demo_min_timesteps = budget(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [run_one_price(args, price, device, output_dir) for price in prices]
    combined = pd.concat(summaries, ignore_index=True)
    combined_path = output_dir / "summary.csv"
    combined.to_csv(combined_path, index=False)

    sac_time = combined.loc[combined["model"] == "M3_SAC", "training_seconds"].mean()
    gasac_time = combined.loc[combined["model"] == "M4_GASAC", "training_seconds"].mean()
    time_saving_pct = (sac_time - gasac_time) / sac_time * 100 if sac_time > 0 else float("nan")

    metadata = {
        "device": device,
        "requested_device": args.device,
        "prices": prices,
        "quick": args.quick,
        "gasac_steps": gasac_steps,
        "sac_steps": sac_steps,
        "demo_min_timesteps": demo_min_timesteps,
        "demo_batch_size": args.demo_batch_size,
        "expert_source": args.expert_source,
        "bc_pretrain_epochs": args.bc_pretrain_epochs,
        "bc_posttrain_epochs": args.bc_posttrain_epochs,
        "bc_charge_loss_weight": args.bc_charge_loss_weight,
        "average_training_time_saving_pct": time_saving_pct,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nsummary_csv={combined_path}")
    print(f"average_training_time_saving_pct={time_saving_pct:.2f}")


if __name__ == "__main__":
    main()

