#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MIMICKIT_DIR = ROOT_DIR / "mimickit"

ALGORITHMS = ("gpm", "sgp", "ewc", "gpm_ref", "sgp_ref", "ewc_ref")


def resolve_repo_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    root_candidate = (ROOT_DIR / path).resolve()
    if root_candidate.exists():
        return str(root_candidate)

    mimickit_candidate = (MIMICKIT_DIR / path).resolve()
    if mimickit_candidate.exists():
        return str(mimickit_candidate)

    return str(root_candidate)


def build_common_args(args):
    common = [
        "--curriculum_config", resolve_repo_path(args.curriculum_config),
        "--rand_seed", str(args.rand_seed),
        "--num_envs", str(args.num_envs),
        "--max_samples", str(args.max_samples),
        "--cl_n_steps", str(args.cl_n_steps),
        "--cl_max_envs", str(args.cl_max_envs),
        "--critic_optimizer", "Adam",
        "--disc_optimizer", "Adam",
    ]
    return common


def build_command(args, algorithm):
    actor_optimizer = "Projection_Adam" if algorithm in ("gpm", "sgp", "gpm_ref", "sgp_ref") else "Adam"
    cmd = [
        sys.executable,
        str(MIMICKIT_DIR / "cl_run.py"),
        *build_common_args(args),
        "--cl_method", algorithm,
        "--actor_optimizer", actor_optimizer,
    ]
    return cmd


def run_one(args, algorithm):
    cmd = build_command(args, algorithm)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MimicKit CL experiments with Python only.")
    parser.add_argument(
        "algorithm",
        choices=ALGORITHMS + ("all",),
        help="Which algorithm to run.",
    )
    parser.add_argument(
        "--curriculum_config",
        default=os.environ.get("CURRICULUM_CONFIG", "data/curriculums/cl_humanoid_example.yaml"),
        help="Curriculum config path relative to mimickit/ or absolute path.",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=int(os.environ.get("RAND_SEED", 42)),
        help="Random seed.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=int(os.environ.get("NUM_ENVS", 64)),
        help="Number of environments.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=int(os.environ.get("MAX_SAMPLES", 1000000)),
        help="Max samples per stage.",
    )
    parser.add_argument(
        "--cl_n_steps",
        type=int,
        default=int(os.environ.get("CL_N_STEPS", 300)),
        help="Number of rollout steps used for CL memory stats.",
    )
    parser.add_argument(
        "--cl_max_envs",
        type=int,
        default=int(os.environ.get("CL_MAX_ENVS", 256)),
        help="Max envs sampled for CL memory stats.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.algorithm == "all":
        for algo in ALGORITHMS:
            run_one(args, algo)
    else:
        run_one(args, args.algorithm)


if __name__ == "__main__":
    main()
