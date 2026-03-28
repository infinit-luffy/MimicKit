"""
Continual Learning Evaluation Script for MimicKit.

Loads a trained CL model and evaluates it on every motion in a curriculum.
Optionally records a video per motion.

Usage:
    python cl_eval.py \
        --curriculum_config data/curriculums/cl_humanoid_example.yaml \
        --model_file output/cl_humanoid/.../model.pt \
        --agent_config data/agents/amp_cl_humanoid_agent.yaml \
        --out_dir output/eval_results \
        --num_episodes 10 \
        --num_envs 16 \
        --video True
"""

import os
import sys
import yaml
import datetime

import util.rsl_rl_util as rsl_rl_util
rsl_rl_util.configure_rsl_rl_path()

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import continual_run as cl_base
import run as run_lib
from util.logger import Logger
import util.mp_util as mp_util

import torch


def run(rank, num_procs, device, master_port, args):
    mp_util.init(rank, num_procs, device, master_port)
    run_lib.set_rand_seed(args)
    run_lib.set_np_formatting()

    curriculum_file = args.parse_string("curriculum_config")
    assert curriculum_file != "", "Missing --curriculum_config"

    model_file = args.parse_string("model_file")
    assert model_file != "", "Missing --model_file"

    curriculum = cl_base.load_curriculum(curriculum_file)
    stages = curriculum["stages"]

    start_stage = args.parse_int("start_stage", 0)
    end_stage   = args.parse_int("end_stage",   len(stages) - 1)
    assert 0 <= start_stage <= end_stage < len(stages)

    num_episodes = args.parse_int("num_episodes", 10)
    num_envs     = args.parse_int("num_envs", 1)
    record_video = args.parse_bool("video", False)

    out_dir = args.parse_string("out_dir", "output/cl_eval")
    cl_base.create_output_dir(out_dir)

    # Build env once (motion will be swapped between stages)
    first_stage = stages[start_stage]
    first_config = cl_base.build_stage_config(start_stage, first_stage, curriculum, args)

    Logger.print("Building environment (record_video={})...".format(record_video))
    env = env_builder.build_env_from_config(
        env_config=first_config["env_config"],
        engine_config=first_config["engine_config"],
        num_envs=num_envs,
        device=device,
        visualize=record_video,
        record_video=record_video,
    )

    results = {}
    stage_names = []

    Logger.print("")
    Logger.print("=" * 60)
    Logger.print(" CL EVALUATION: {} motions, {} episodes each".format(
        end_stage - start_stage + 1, num_episodes))
    Logger.print("=" * 60)

    for task_id in range(start_stage, end_stage + 1):
        stage = stages[task_id]
        stage_config = cl_base.build_stage_config(task_id, stage, curriculum, args)
        stage_name = stage_config["name"]
        stage_names.append(stage_name)

        Logger.print("")
        Logger.print("--- Task {} / {} : {} ---".format(
            task_id - start_stage + 1, end_stage - start_stage + 1, stage_name))

        # Swap motion
        motion_file = stage_config["env_config"]["motion_file"]
        env._load_motions(motion_file)

        # Build agent and load model
        agent = agent_builder.build_agent_from_config(
            stage_config["agent_config"], env, device
        )
        agent.load(model_file)
        agent.set_current_motion(task_id)

        # Evaluate
        result = agent.test_model(num_episodes=num_episodes)

        mean_return = float(result.get("mean_return", 0.0))
        mean_ep_len = float(result.get("mean_ep_len", 0.0))
        results[stage_name] = {
            "task_id": task_id,
            "mean_return": mean_return,
            "mean_ep_len": mean_ep_len,
            "num_episodes": int(result.get("num_eps", num_episodes)),
        }

        Logger.print("  mean_return  = {:.3f}".format(mean_return))
        Logger.print("  mean_ep_len  = {:.1f}".format(mean_ep_len))

        del agent
        torch.cuda.empty_cache()

    # Print summary table
    Logger.print("")
    Logger.print("=" * 60)
    Logger.print(" EVALUATION SUMMARY")
    Logger.print("=" * 60)
    Logger.print("{:<16s}  {:>12s}  {:>12s}".format("motion", "mean_return", "mean_ep_len"))
    Logger.print("-" * 44)
    for name in stage_names:
        r = results[name]
        Logger.print("{:<16s}  {:>12.3f}  {:>12.1f}".format(
            name[:16], r["mean_return"], r["mean_ep_len"]))

    # Average performance
    ap_return = sum(r["mean_return"] for r in results.values()) / max(len(results), 1)
    ap_ep_len = sum(r["mean_ep_len"] for r in results.values()) / max(len(results), 1)
    Logger.print("-" * 44)
    Logger.print("{:<16s}  {:>12.3f}  {:>12.1f}".format("AVERAGE", ap_return, ap_ep_len))
    Logger.print("=" * 60)

    # Save results
    if mp_util.is_root_proc():
        save_data = {
            "model_file": model_file,
            "curriculum_file": curriculum_file,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_episodes": num_episodes,
            "average_return": float(ap_return),
            "average_ep_len": float(ap_ep_len),
            "per_task": {
                name: {k: float(v) if isinstance(v, float) else v
                       for k, v in r.items()}
                for name, r in results.items()
            },
        }
        out_file = os.path.join(out_dir, "eval_results.yaml")
        with open(out_file, "w") as f:
            yaml.safe_dump(save_data, f, sort_keys=False)
        Logger.print("Results saved to {}".format(out_file))

    return


def main(argv):
    args = cl_base.load_args(argv)
    master_port = args.parse_int("master_port", None)
    devices = args.parse_strings("devices", ["cuda:0"])

    import numpy as np
    if master_port is None:
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for rank in range(1, len(devices)):
        proc = torch.multiprocessing.Process(
            target=run,
            args=[rank, len(devices), devices[rank], master_port, args]
        )
        proc.start()
        processes.append(proc)

    run(0, len(devices), devices[0], master_port, args)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main(sys.argv)
