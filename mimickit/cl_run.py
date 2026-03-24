"""
Continual Learning Training Script for MimicKit.

Orchestrates sequential motion learning with CBP + SGP protection.
Extends the continual_run.py patterns with SGP memory extraction/injection
between stages.

Usage:
    python cl_run.py --curriculum_config data/curriculums/cl_humanoid_example.yaml

Each stage in the curriculum trains one motion. Between stages:
- SGP features are extracted via SVD on actor activations
- Projection matrices protect previous motion subspaces
- Discriminator is reset for the new motion
- CBP utility tracking is reset
"""

import copy
import datetime
import numpy as np
import os
import sys
import time
import yaml

import util.rsl_rl_util as rsl_rl_util
rsl_rl_util.configure_rsl_rl_path()

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.cl.sgp_memory as sgp_memory
import continual_run as cl_base
import run as run_lib
from util.logger import Logger
import util.mp_util as mp_util

import torch


def build_run_dir(base_out_dir, agent_name, rand_seed):
    """Build output directory name: base/YYYY-MM-DD_seedXXX_agentname/"""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    seed_str = "seed{}".format(rand_seed)
    run_name = "{}_{}_{}" .format(date_str, seed_str, agent_name)
    return os.path.join(base_out_dir, run_name)


def resolve_rand_seed(args):
    """Resolve the random seed from args, or generate one from time."""
    if args.has_key("rand_seed"):
        return args.parse_int("rand_seed")
    else:
        return int(np.uint64(time.time() * 256))


def save_training_config(out_dir, curriculum, curriculum_file, args, rand_seed):
    """Save full training configuration to the output directory."""
    if not mp_util.is_root_proc():
        return

    config = {
        "curriculum_file": curriculum_file,
        "rand_seed": int(rand_seed),
        "timestamp": datetime.datetime.now().isoformat(),
        "out_dir": out_dir,
        "start_stage": args.parse_int("start_stage", 0),
        "end_stage": args.parse_int("end_stage", len(curriculum["stages"]) - 1),
        "num_stages": len(curriculum["stages"]),
        "stage_names": [s.get("name", "stage_{}".format(i))
                        for i, s in enumerate(curriculum["stages"])],
        "env_config": curriculum.get("env_config", ""),
        "agent_config": curriculum.get("agent_config", ""),
        "engine_config": curriculum.get("engine_config", ""),
        "max_samples_per_stage": curriculum.get("max_samples", "default"),
        "num_envs": curriculum.get("num_envs", "default"),
        "logger": curriculum.get("logger", "tb"),
    }

    config_path = os.path.join(out_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    Logger.print("Training config saved to {}".format(config_path))


def save_evaluation_results(out_dir, results):
    """Save evaluation results to YAML."""
    if not mp_util.is_root_proc():
        return

    # Convert tensors to floats for serialization
    serializable = {}
    for name, result in results.items():
        serializable[name] = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in result.items()
        }

    eval_path = os.path.join(out_dir, "evaluation_results.yaml")
    with open(eval_path, "w") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)
    Logger.print("Evaluation results saved to {}".format(eval_path))


def collect_obs_for_sgp(env, agent, n_steps=20):
    """Run trained policy and collect observations for SGP feature extraction.

    Returns observations concatenated with motion one-hot, matching actor input.
    """
    agent.eval()
    obs, info = env.reset()
    obs_list = []

    with torch.no_grad():
        for _ in range(n_steps):
            action, action_info = agent._decide_action(obs, info)
            norm_obs = agent._obs_norm.normalize(obs)
            motion_onehot = agent._motion_onehot_buf
            actor_input = torch.cat([norm_obs, motion_onehot], dim=-1)
            obs_list.append(actor_input.clone())

            next_obs, r, done, next_info = env.step(action)
            obs = next_obs
            info = next_info

    return torch.cat(obs_list, dim=0)


def evaluate_motion(agent, env, motion_id, num_episodes=10):
    """Test a specific motion by setting the motion ID and running test episodes."""
    agent.set_current_motion(motion_id)
    result = agent.test_model(num_episodes=num_episodes)
    return result


def train_cl_stage(stage_config, task_id, device, in_model_file, sgp_mem,
                   prev_signature, curriculum_file):
    """Train one CL stage (one motion).

    Args:
        stage_config: Stage configuration dict from build_stage_config.
        task_id: Integer task ID for this motion.
        device: Torch device.
        in_model_file: Path to previous stage's model checkpoint.
        sgp_mem: SGPMemoryBank instance.
        prev_signature: Previous env signature for compatibility check.
        curriculum_file: Path to curriculum config file.

    Returns:
        (out_model_file, curr_signature, stage_record)
    """
    stage_name = stage_config["name"]
    Logger.print("")
    Logger.print("=" * 60)
    Logger.print(" CL STAGE {}: {} (task_id={})".format(
        stage_config["index"], stage_name, task_id))
    Logger.print("=" * 60)

    cl_base.save_stage_files(stage_config, curriculum_file=curriculum_file,
                             in_model_file=in_model_file)

    # A. Build environment
    env = env_builder.build_env_from_config(
        env_config=stage_config["env_config"],
        engine_config=stage_config["engine_config"],
        num_envs=stage_config["num_envs"],
        device=device,
        visualize=stage_config["visualize"],
        record_video=stage_config["record_video"]
    )

    # B. Build agent (AMP_CL)
    agent = agent_builder.build_agent_from_config(
        stage_config["agent_config"], env, device
    )

    # Verify env compatibility
    curr_signature = cl_base.calc_env_signature(env)
    if prev_signature is not None:
        assert curr_signature == prev_signature, \
            "Stage {} changed obs/action spaces: {} vs {}".format(
                stage_name, curr_signature, prev_signature)

    # C. Load previous model + inject CL state
    if task_id > 0 and in_model_file != "":
        Logger.print("Loading previous model from {}".format(in_model_file))
        agent.load(in_model_file)

        # Build and inject SGP projection matrices
        Logger.print("Building SGP projection matrices from {} tasks...".format(
            len(sgp_mem.memory_bank)))
        feature_mats = sgp_mem.build_projection_matrices()
        anchors = sgp_mem.build_anchor_weights(
            agent._model._actor_layers, feature_mats
        )
        agent.prepare_for_new_task(task_id, feature_mats, anchors)
    elif task_id == 0:
        # First task: optionally load a pretrained base model
        model_file = stage_config.get("model_file", "")
        if model_file != "":
            agent.load(model_file)
        agent.set_current_motion(task_id)

    # D. Train
    agent.train_model(
        max_samples=stage_config["max_samples"],
        out_dir=stage_config["out_dir"],
        save_int_models=stage_config["save_int_models"],
        logger_type=stage_config["logger_type"]
    )

    # E. Extract SGP features for memory bank
    Logger.print("Extracting SGP features for task {} ({})...".format(task_id, stage_name))
    obs_data = collect_obs_for_sgp(env, agent, n_steps=20)
    new_features = sgp_mem.extract_features(agent._model._actor_layers, obs_data)
    sgp_mem.memory_bank.append(new_features)
    Logger.print("Memory bank updated. Total tasks remembered: {}".format(
        len(sgp_mem.memory_bank)))

    # F. Save model and memory
    out_model_file = os.path.join(stage_config["out_dir"], "model.pt")
    sgp_memory_file = os.path.join(stage_config["out_dir"], "sgp_memory.pt")
    agent.save(out_model_file)
    sgp_mem.save(sgp_memory_file)

    stage_record = {
        "index": stage_config["index"],
        "name": stage_name,
        "task_id": task_id,
        "out_dir": stage_config["out_dir"],
        "model_file": out_model_file,
        "sgp_memory_file": sgp_memory_file,
        "init_model_file": in_model_file,
        "max_samples": int(stage_config["max_samples"])
    }

    # Cleanup
    del env, agent
    torch.cuda.empty_cache()

    return out_model_file, curr_signature, stage_record


def evaluate_all_motions(stages, final_model_file, sgp_mem, device,
                         curriculum, args):
    """Evaluate retention on all learned motions using the final model."""
    Logger.print("")
    Logger.print("=" * 60)
    Logger.print(" EVALUATION: Testing all {} motions".format(len(stages)))
    Logger.print("=" * 60)

    results = {}
    for task_id, stage in enumerate(stages):
        stage_config = cl_base.build_stage_config(task_id, stage, curriculum, args)
        stage_name = stage_config["name"]

        env = env_builder.build_env_from_config(
            env_config=stage_config["env_config"],
            engine_config=stage_config["engine_config"],
            num_envs=stage_config["num_envs"],
            device=device,
            visualize=False,
            record_video=False
        )

        agent = agent_builder.build_agent_from_config(
            stage_config["agent_config"], env, device
        )
        agent.load(final_model_file)

        result = evaluate_motion(agent, env, task_id, num_episodes=10)
        results[stage_name] = result

        Logger.print("  Task {} ({}): mean_return={:.2f}, mean_ep_len={:.2f}".format(
            task_id, stage_name, result["mean_return"], result["mean_ep_len"]))

        del env, agent
        torch.cuda.empty_cache()

    return results


def run(rank, num_procs, device, master_port, args, rand_seed):
    curriculum_config_file = args.parse_string("curriculum_config")
    assert curriculum_config_file != "", "Missing --curriculum_config"

    curriculum = cl_base.load_curriculum(curriculum_config_file)
    stages = curriculum["stages"]

    start_stage = args.parse_int("start_stage", 0)
    end_stage = args.parse_int("end_stage", len(stages) - 1)
    assert 0 <= start_stage < len(stages)
    assert start_stage <= end_stage < len(stages)

    # Build output dir: base/YYYY-MM-DD_seedXXX_agentname/
    agent_file = curriculum.get("agent_config", args.parse_string("agent_config", ""))
    agent_name = "AMP_CL"
    if agent_file != "":
        agent_cfg = cl_base.load_yaml(agent_file)
        agent_name = agent_cfg.get("agent_name", "AMP_CL")

    base_out_dir = curriculum.get("out_dir",
                                   args.parse_string("out_dir", "output/"))
    curriculum_out_dir = build_run_dir(base_out_dir, agent_name, rand_seed)

    mp_util.init(rank, num_procs, device, master_port)
    run_lib.set_rand_seed(args)
    run_lib.set_np_formatting()
    cl_base.create_output_dir(curriculum_out_dir)

    # Save full training config
    save_training_config(curriculum_out_dir, curriculum,
                         curriculum_config_file, args, rand_seed)

    in_model_file = args.parse_string("model_file",
                                       curriculum.get("model_file", ""))
    prev_signature = None
    stage_records = []

    # Load existing SGP memory if resuming
    sgp_mem = sgp_memory.SGPMemoryBank(device=device)
    sgp_memory_file = args.parse_string("sgp_memory_file", "")
    if sgp_memory_file != "":
        sgp_mem.load(sgp_memory_file)
        Logger.print("Loaded SGP memory from {} ({} tasks)".format(
            sgp_memory_file, len(sgp_mem.memory_bank)))

    # Override stage out_dirs to go under curriculum_out_dir
    # Train each motion stage sequentially
    for stage_idx in range(start_stage, end_stage + 1):
        stage = stages[stage_idx]
        stage_config = cl_base.build_stage_config(stage_idx, stage, curriculum, args)
        task_id = stage_idx  # task_id = stage index

        # Redirect stage output into the run directory
        stage_name = cl_base.sanitize_stage_name(stage_config["name"])
        stage_config["out_dir"] = os.path.join(
            curriculum_out_dir, "stage_{:02d}_{}".format(stage_idx, stage_name)
        )

        out_model_file, prev_signature, stage_record = train_cl_stage(
            stage_config=stage_config,
            task_id=task_id,
            device=device,
            in_model_file=in_model_file,
            sgp_mem=sgp_mem,
            prev_signature=prev_signature,
            curriculum_file=curriculum_config_file
        )

        in_model_file = out_model_file
        stage_records.append(stage_record)

    # Final evaluation
    skip_eval = args.parse_bool("skip_eval", False)
    eval_results = None
    if not skip_eval:
        eval_results = evaluate_all_motions(
            stages[:end_stage + 1], in_model_file,
            sgp_mem, device, curriculum, args
        )
        save_evaluation_results(curriculum_out_dir, eval_results)

    # Finalize
    cl_base.finalize_curriculum_outputs(curriculum_out_dir, stage_records,
                                        in_model_file)

    # Save final SGP memory
    final_sgp_path = os.path.join(curriculum_out_dir, "sgp_memory_final.pt")
    sgp_mem.save(final_sgp_path)
    Logger.print("Final SGP memory saved to {}".format(final_sgp_path))
    Logger.print("")
    Logger.print("All outputs saved to: {}".format(curriculum_out_dir))

    return


def main(argv):
    root_rank = 0
    args = cl_base.load_args(argv)
    master_port = args.parse_int("master_port", None)
    devices = args.parse_strings("devices", ["cuda:0"])

    num_workers = len(devices)
    assert num_workers > 0

    if master_port is None:
        master_port = np.random.randint(6000, 7000)

    # Resolve seed early so all workers share the same run directory name
    rand_seed = resolve_rand_seed(args)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for rank in range(1, num_workers):
        curr_device = devices[rank]
        proc = torch.multiprocessing.Process(
            target=run,
            args=[rank, num_workers, curr_device, master_port, args, rand_seed]
        )
        proc.start()
        processes.append(proc)

    root_device = devices[0]
    run(root_rank, num_workers, root_device, master_port, args, rand_seed)

    for proc in processes:
        proc.join()

    return


if __name__ == "__main__":
    main(sys.argv)
