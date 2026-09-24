#!/usr/bin/env python3
"""Inspect, dry-run, or explicitly launch legal-landscape training."""

from __future__ import annotations

import argparse
import json

from legal_landscape.config import load_config, parse_overrides
from legal_landscape.training.train import run_dummy_train_step, run_real_training, training_plan


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/model/qwen35_9b_qlora.yaml")
    result.add_argument("--experiment", default="M")
    result.add_argument("--train-data")
    result.add_argument("--counterfactual-data")
    result.add_argument("--evaluation-data")
    result.add_argument("--prediction-dir")
    result.add_argument("--output-dir", default="outputs/training")
    result.add_argument("--limit", type=int, default=8)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--dummy", action="store_true", help="Run one CPU dummy optimizer step")
    result.add_argument("--resume-from-checkpoint")
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config, overrides=parse_overrides(args.set))
    plan = training_plan(config, experiment=args.experiment)
    plan.update(
        {
            "limit": args.limit,
            "output_dir": args.output_dir,
            "prediction_dir": args.prediction_dir,
            "evaluation_data": args.evaluation_data,
            "resume": args.resume_from_checkpoint,
        }
    )
    if args.dummy:
        if not args.execute:
            print(
                json.dumps({**plan, "dry_run": True, "dummy": True}, ensure_ascii=False, indent=2)
            )
            return 0
        value = run_dummy_train_step(seed=plan["seed"])
        print(json.dumps({**plan, "dummy_loss": value}, ensure_ascii=False, indent=2))
        return 0
    if args.dry_run or not args.execute:
        print(json.dumps({**plan, "dry_run": True}, ensure_ascii=False, indent=2))
        return 0
    if not args.train_data:
        raise SystemExit("--train-data is required with --execute")
    result = run_real_training(
        config,
        train_data=args.train_data,
        counterfactual_data=args.counterfactual_data,
        evaluation_data=args.evaluation_data,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment,
        limit=args.limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
