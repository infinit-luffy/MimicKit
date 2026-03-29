"""
Continual Learning Training Script for MimicKit.

Orchestrates sequential motion learning with gradient projection (GPM/SGP)
or EWC protection between stages.

Usage:
    python cl_run.py --curriculum_config data/curriculums/cl_humanoid_example.yaml

Each stage in the curriculum trains one motion. Between stages:
- GPM/SGP: features are extracted via SVD on actor activations, projection
  matrices protect previous motion subspaces.
- EWC: Fisher Information is computed, quadratic penalty protects parameters.
- Discriminator is reset for the new motion.
- CBP utility tracking is reset.
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
import learning.cl.projection_memory as projection_memory
import learning.cl.ewc_memory as ewc_memory
import learning.cl.projection_memory_ref as projection_memory_ref
import learning.cl.ewc_memory_ref as ewc_memory_ref
import continual_run as cl_base
import run as run_lib
from util.logger import Logger
import util.mp_util as mp_util

import torch


PROJECTION_METHODS = ("gpm", "sgp", "gpm_ref", "sgp_ref")
EWC_METHODS = ("ewc", "ewc_ref")


def _flatten_tensor_stats(tensors):
    valid = [t.detach().float().reshape(-1) for t in tensors if t is not None and t.numel() > 0]
    if not valid:
        return {
            "num_tensors": 0,
            "num_values": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "l1": 0.0,
            "l2": 0.0,
        }

    flat = torch.cat(valid, dim=0)
    return {
        "num_tensors": len(valid),
        "num_values": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l1": float(flat.abs().sum().item()),
        "l2": float(flat.norm().item()),
    }


def summarize_projection_memory(cl_mem):
    layer_stats = []
    valid_layers = 0
    total_rank = 0
    total_dim = 0

    for idx, U in enumerate(getattr(cl_mem, "feature_list", [])):
        if U is None:
            layer_stats.append({"layer": idx, "rank": 0, "feat_dim": 0, "rank_ratio": 0.0})
            continue

        rank = int(U.shape[1])
        feat_dim = int(U.shape[0])
        valid_layers += 1
        total_rank += rank
        total_dim += feat_dim

        layer_info = {
            "layer": idx,
            "rank": rank,
            "feat_dim": feat_dim,
            "rank_ratio": float(rank / max(feat_dim, 1)),
        }

        if hasattr(cl_mem, "importance_list") and idx < len(cl_mem.importance_list):
            alpha = cl_mem.importance_list[idx]
            if alpha is not None and alpha.numel() > 0:
                alpha = alpha.detach().float()
                layer_info.update({
                    "alpha_mean": float(alpha.mean().item()),
                    "alpha_std": float(alpha.std(unbiased=False).item()),
                    "alpha_min": float(alpha.min().item()),
                    "alpha_max": float(alpha.max().item()),
                })

        layer_stats.append(layer_info)

    return {
        "type": type(cl_mem).__name__,
        "method": getattr(cl_mem, "method", None),
        "num_tasks": int(getattr(cl_mem, "num_tasks", 0)),
        "num_layers": len(getattr(cl_mem, "feature_list", [])),
        "valid_layers": valid_layers,
        "total_rank": total_rank,
        "total_dim": total_dim,
        "mean_rank_ratio": float(total_rank / max(total_dim, 1)),
        "alpha_stats": _flatten_tensor_stats(
            [imp for imp in getattr(cl_mem, "importance_list", []) if imp is not None]
        ),
        "layers": layer_stats,
    }


def summarize_projection_data(proj_data):
    layer_stats = []
    valid_layers = 0
    total_rank = 0
    total_dim = 0

    for idx, data in enumerate(proj_data):
        if data is None:
            layer_stats.append({"layer": idx, "rank": 0, "feat_dim": 0, "rank_ratio": 0.0})
            continue

        valid_layers += 1
        if isinstance(data, dict):
            U = data["U"]
            alpha = data["alpha"].detach().float()
            rank = int(U.shape[1])
            feat_dim = int(U.shape[0])
            layer_stats.append({
                "layer": idx,
                "rank": rank,
                "feat_dim": feat_dim,
                "rank_ratio": float(rank / max(feat_dim, 1)),
                "alpha_mean": float(alpha.mean().item()),
                "alpha_std": float(alpha.std(unbiased=False).item()),
                "alpha_min": float(alpha.min().item()),
                "alpha_max": float(alpha.max().item()),
            })
        else:
            P = data.detach().float()
            feat_dim = int(P.shape[0])
            diag = P.diagonal()
            rank = int((diag.abs() > 1e-6).sum().item())
            layer_stats.append({
                "layer": idx,
                "rank": rank,
                "feat_dim": feat_dim,
                "rank_ratio": float(rank / max(feat_dim, 1)),
                "proj_trace": float(diag.sum().item()),
            })

        total_rank += rank
        total_dim += feat_dim

    return {
        "valid_layers": valid_layers,
        "total_rank": total_rank,
        "total_dim": total_dim,
        "mean_rank_ratio": float(total_rank / max(total_dim, 1)),
        "layers": layer_stats,
    }


def summarize_fisher(fisher):
    stats = _flatten_tensor_stats(list(fisher.values()))
    stats["num_params"] = len(fisher)
    return stats


def summarize_ewc_memory(cl_mem):
    if getattr(cl_mem, "online", False):
        fisher = getattr(cl_mem, "_online_fisher", {})
        params = getattr(cl_mem, "_online_params", {})
        return {
            "type": type(cl_mem).__name__,
            "online": True,
            "num_tasks": int(getattr(cl_mem, "_task_count", 1 if fisher else 0)),
            "num_params": len(params),
            "fisher_stats": summarize_fisher(fisher),
        }

    tasks = getattr(cl_mem, "tasks", [])
    fisher_tensors = []
    for task in tasks:
        fisher_tensors.extend(task.get("fisher", {}).values())

    return {
        "type": type(cl_mem).__name__,
        "online": False,
        "num_tasks": len(tasks),
        "num_params_last_task": len(tasks[-1]["params"]) if tasks else 0,
        "fisher_stats_all_tasks": _flatten_tensor_stats(fisher_tensors),
    }


def extract_metric_row(eval_row, metric_key):
    return {
        task_name: float(task_metrics.get(metric_key, 0.0))
        for task_name, task_metrics in eval_row.items()
    }


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


def collect_cli_overrides(args):
    overrides = {}

    int_args = ["rand_seed", "start_stage", "end_stage", "num_envs", "max_samples"]
    for key in int_args:
        if args.has_key(key):
            overrides[key] = args.parse_int(key)

    string_args = [
        "curriculum_config", "env_config", "agent_config", "engine_config",
        "out_dir", "logger", "model_file", "cl_memory_file", "sgp_memory_file",
        "cl_method", "algorithm", "algo", "optimizer", "optim",
        "actor_optimizer", "actor_optim",
        "critic_optimizer", "critic_optim",
        "disc_optimizer", "disc_optim",
    ]
    for key in string_args:
        if args.has_key(key):
            overrides[key] = args.parse_string(key)

    return overrides


def save_training_config(out_dir, curriculum, curriculum_file, args, rand_seed,
                         first_stage_config):
    """Save full training configuration to the output directory."""
    if not mp_util.is_root_proc():
        return

    agent_config = first_stage_config["agent_config"]
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
        "env_config": args.parse_string("env_config", curriculum.get("env_config", "")),
        "agent_config": args.parse_string("agent_config", curriculum.get("agent_config", "")),
        "engine_config": args.parse_string("engine_config", curriculum.get("engine_config", "")),
        "max_samples_per_stage": args.parse_int(
            "max_samples", curriculum.get("max_samples", "default")
        ),
        "num_envs": args.parse_int("num_envs", curriculum.get("num_envs", "default")),
        "logger": args.parse_string("logger", curriculum.get("logger", "tb")),
        "cl_method": agent_config.get("cl_method", "gpm"),
        "actor_optimizer": agent_config.get("actor_optimizer", {}).get("type", "default"),
        "critic_optimizer": agent_config.get("critic_optimizer", {}).get("type", "default"),
        "disc_optimizer": agent_config.get("disc_optimizer", {}).get("type", "default"),
        "command_line_overrides": collect_cli_overrides(args),
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


def evaluate_after_stage(env, stages, current_stage_idx, model_file, device,
                         curriculum, args, num_episodes=10):
    """Evaluate all learned tasks after completing a training stage.

    Returns a dict mapping task names to a small metrics dict.
    """
    Logger.print("")
    Logger.print("--- Eval after stage {} ({}) on tasks 0..{} ---".format(
        current_stage_idx, stages[current_stage_idx].get("name", ""),
        current_stage_idx))

    row = {}
    for task_id in range(current_stage_idx + 1):
        stage = stages[task_id]
        stage_config = cl_base.build_stage_config(task_id, stage, curriculum, args)
        stage_name = stage_config["name"]

        motion_file = stage_config["env_config"]["motion_file"]
        env._load_motions(motion_file)

        agent = agent_builder.build_agent_from_config(
            stage_config["agent_config"], env, device
        )
        agent.load(model_file)

        result = evaluate_motion(agent, env, task_id, num_episodes=num_episodes)
        mean_return = float(result.get("mean_return", 0.0))
        ep_len = float(result.get("mean_ep_len", 0.0))
        row[stage_name] = {
            "mean_return": mean_return,
            "mean_ep_len": ep_len,
        }

        Logger.print("  task {} ({}): mean_return={:.2f}, mean_ep_len={:.1f}".format(
            task_id, stage_name, mean_return, ep_len))

        del agent
        torch.cuda.empty_cache()

    return row


def compute_cl_metrics(perf_matrix, stage_names):
    """Compute standard CL metrics from one scalar performance matrix.

    Args:
        perf_matrix: list[dict], where perf_matrix[i][name] is a scalar metric
                     after training stage i on task 'name'.
        stage_names: list of stage name strings in order.

    Returns:
        dict with AP, BWT, Forgetting metrics.
    """
    K = len(perf_matrix) - 1  # last training stage index
    if K < 0:
        return {}

    last_row = perf_matrix[K]

    # AP: Average Performance on all tasks using final model
    ap_values = [last_row.get(name, 0.0) for name in stage_names[:K + 1]]
    ap = sum(ap_values) / len(ap_values) if ap_values else 0.0

    # BWT: Backward Transfer = mean(R[K,j] - R[j,j]) for j < K
    bwt_values = []
    for j in range(K):
        name = stage_names[j]
        r_kj = last_row.get(name, 0.0)
        r_jj = perf_matrix[j].get(name, 0.0)
        bwt_values.append(r_kj - r_jj)
    bwt = sum(bwt_values) / len(bwt_values) if bwt_values else 0.0

    # Forgetting: mean(max_i(R[i,j]) - R[K,j]) for j < K
    fgt_values = []
    for j in range(K):
        name = stage_names[j]
        best_rij = max(perf_matrix[i].get(name, 0.0)
                       for i in range(j, K + 1))
        r_kj = last_row.get(name, 0.0)
        fgt_values.append(best_rij - r_kj)
    fgt = sum(fgt_values) / len(fgt_values) if fgt_values else 0.0

    metrics = {
        "average_performance": ap,
        "backward_transfer": bwt,
        "forgetting": fgt,
    }
    return metrics


def print_performance_matrix(perf_matrix, stage_names, metric_name="metric"):
    """Pretty-print one scalar performance matrix."""
    Logger.print("")
    Logger.print("=" * 60)
    Logger.print(" PERFORMANCE MATRIX ({})".format(metric_name))
    Logger.print("=" * 60)

    # Header
    header = "{:<12s}".format("trained\\eval")
    for name in stage_names:
        header += " {:>10s}".format(name[:10])
    Logger.print(header)

    # Rows
    for i, row in enumerate(perf_matrix):
        line = "{:<12s}".format(stage_names[i][:12])
        for j, name in enumerate(stage_names):
            if name in row:
                line += " {:>10.1f}".format(row[name])
            else:
                line += " {:>10s}".format("-")
        Logger.print(line)

    Logger.print("")


def collect_obs_for_projection(env, agent, n_steps=300, max_envs=256):
    """Run trained policy and collect observations for feature extraction.

    Returns observations concatenated with motion one-hot, matching actor input.

    Args:
        n_steps: Number of rollout steps.
        max_envs: Max environments to sample from per step. If the env has more
                  environments, only the first max_envs are used. This bounds
                  memory and SVD cost regardless of training num_envs.
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
            obs_list.append(actor_input[:max_envs].clone())

            next_obs, r, done, next_info = env.step(action)
            obs = next_obs
            info = next_info

    return torch.cat(obs_list, dim=0)


def evaluate_motion(agent, env, motion_id, num_episodes=10):
    """Test a specific motion by setting the motion ID and running test episodes."""
    agent.set_current_motion(motion_id)
    result = agent.test_model(num_episodes=num_episodes)
    return result


def train_cl_stage(env, stage_config, task_id, device, in_model_file, cl_mem,
                   curriculum_file, cl_method="gpm", cl_n_steps=300, cl_max_envs=32):
    """Train one CL stage (one motion), reusing the existing env.

    Args:
        env: Reusable environment instance (motion is swapped, not recreated).
        stage_config: Stage configuration dict from build_stage_config.
        task_id: Integer task ID for this motion.
        device: Torch device.
        in_model_file: Path to previous stage's model checkpoint.
        cl_mem: ProjectionMemoryBank or EWCMemory instance.
        curriculum_file: Path to curriculum config file.
        cl_method: "gpm", "sgp", or "ewc".

    Returns:
        (out_model_file, stage_record)
    """
    stage_name = stage_config["name"]
    Logger.print("")
    Logger.print("=" * 60)
    Logger.print(" CL STAGE {}: {} (task_id={}, method={})".format(
        stage_config["index"], stage_name, task_id, cl_method))
    Logger.print("=" * 60)

    cl_base.save_stage_files(stage_config, curriculum_file=curriculum_file,
                             in_model_file=in_model_file)

    # A. Swap motion file on the existing env (no engine recreation)
    motion_file = stage_config["env_config"]["motion_file"]
    Logger.print("Loading motion: {}".format(motion_file))
    env._load_motions(motion_file)

    # B. Build agent (AMP_CL)
    agent = agent_builder.build_agent_from_config(
        stage_config["agent_config"], env, device
    )
    proj_summary_before = None
    anchor_summary_before = None
    memory_summary_after = None
    fisher_summary = None
    ewc_summary_after = None

    # C. Load previous model + inject CL state
    if task_id > 0 and in_model_file != "":
        Logger.print("Loading previous model from {}".format(in_model_file))
        agent.load(in_model_file)

        if cl_method in PROJECTION_METHODS:
            # Build and inject projection matrices
            Logger.print("Building {} projection matrices from {} tasks...".format(
                cl_method.upper(), cl_mem.num_tasks))
            proj_data = cl_mem.build_projection_matrices()
            anchors = cl_mem.build_anchor_weights(
                agent._model._actor_layers, proj_data,
                output_layer=agent._model._action_dist
            )
            proj_summary_before = summarize_projection_data(proj_data)
            anchor_summary_before = {
                "num_anchors": len(anchors),
                "anchor_stats": _flatten_tensor_stats([a for a in anchors if a is not None]),
            }
            agent.prepare_for_new_task(task_id, projection_data=proj_data,
                                        projection_anchors=anchors)
        elif cl_method in EWC_METHODS:
            agent.prepare_for_new_task(task_id, ewc_memory=cl_mem)

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

    # E. Extract features / compute Fisher for memory
    if cl_method in PROJECTION_METHODS:
        Logger.print("Updating {} memory for task {} ({})...".format(
            cl_method.upper(), task_id, stage_name))
        obs_data = collect_obs_for_projection(env, agent, n_steps=cl_n_steps,
                                              max_envs=cl_max_envs)
        cl_mem.update_memory(
            agent._model._actor_layers, obs_data, task_id,
            output_layer=agent._model._action_dist
        )
        memory_summary_after = summarize_projection_memory(cl_mem)
        Logger.print("Projection summary: total_rank={}, mean_rank_ratio={:.4f}".format(
            memory_summary_after["total_rank"], memory_summary_after["mean_rank_ratio"]))
        Logger.print("Projection memory updated. Total tasks: {}".format(
            cl_mem.num_tasks))
    elif cl_method in EWC_METHODS:
        Logger.print("Computing Fisher for task {} ({})...".format(task_id, stage_name))
        fisher = cl_mem.compute_fisher(agent, env, n_steps=cl_n_steps)
        fisher_summary = summarize_fisher(fisher)
        cl_mem.register_task(agent, fisher)
        ewc_summary_after = summarize_ewc_memory(cl_mem)
        Logger.print("Fisher summary: mean={:.6f}, max={:.6f}, l2={:.6f}".format(
            fisher_summary["mean"], fisher_summary["max"], fisher_summary["l2"]))

    # F. Save model and memory
    out_model_file = os.path.join(stage_config["out_dir"], "model.pt")
    cl_memory_file = os.path.join(stage_config["out_dir"], "cl_memory.pt")
    agent.save(out_model_file)
    cl_mem.save(cl_memory_file)

    stage_record = {
        "index": stage_config["index"],
        "name": stage_name,
        "task_id": task_id,
        "out_dir": stage_config["out_dir"],
        "model_file": out_model_file,
        "cl_memory_file": cl_memory_file,
        "init_model_file": in_model_file,
        "max_samples": int(stage_config["max_samples"]),
        "cl_method": cl_method,
        "projection_summary_before": proj_summary_before,
        "anchor_summary_before": anchor_summary_before,
        "projection_memory_summary_after": memory_summary_after,
        "fisher_summary": fisher_summary,
        "ewc_summary_after": ewc_summary_after,
    }
    cl_base.save_yaml(stage_record, os.path.join(stage_config["out_dir"], "stage_metrics.yaml"))

    # Cleanup agent only (env is reused)
    del agent
    torch.cuda.empty_cache()

    return out_model_file, stage_record


def evaluate_all_motions(env, stages, final_model_file, device,
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

        # Swap motion on the shared env
        motion_file = stage_config["env_config"]["motion_file"]
        env._load_motions(motion_file)

        agent = agent_builder.build_agent_from_config(
            stage_config["agent_config"], env, device
        )
        agent.load(final_model_file)

        result = evaluate_motion(agent, env, task_id, num_episodes=10)
        results[stage_name] = result

        Logger.print("  Task {} ({}): mean_return={:.2f}, mean_ep_len={:.2f}".format(
            task_id, stage_name, result["mean_return"], result["mean_ep_len"]))

        del agent
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

    first_stage = stages[start_stage]
    first_stage_config = cl_base.build_stage_config(
        start_stage, first_stage, curriculum, args
    )

    # Build output dir: base/YYYY-MM-DD_seedXXX_agentname/
    agent_name = first_stage_config["agent_config"].get("agent_name", "AMP_CL")

    base_out_dir = curriculum.get("out_dir",
                                   args.parse_string("out_dir", "output/"))
    curriculum_out_dir = build_run_dir(base_out_dir, agent_name, rand_seed)

    mp_util.init(rank, num_procs, device, master_port)
    run_lib.set_rand_seed(args)
    run_lib.set_np_formatting()
    cl_base.create_output_dir(curriculum_out_dir)

    # Save full training config
    save_training_config(curriculum_out_dir, curriculum,
                         curriculum_config_file, args, rand_seed,
                         first_stage_config)

    in_model_file = args.parse_string("model_file",
                                       curriculum.get("model_file", ""))
    prev_signature = None
    stage_records = []

    # Determine CL method from the effective agent config after CLI overrides.
    agent_cfg = first_stage_config["agent_config"]
    cl_method = agent_cfg.get("cl_method", "gpm")

    # Create CL memory
    # CLI args override yaml: --threshold, --threshold_inc, --scale_coff, --gpm_mini_batch
    if cl_method in PROJECTION_METHODS:
        threshold = args.parse_float(
            "threshold",
            agent_cfg.get("projection_threshold", agent_cfg.get("sgp_threshold", 0.98))
        )
        threshold_inc = args.parse_float(
            "threshold_inc",
            agent_cfg.get("projection_threshold_inc", 0.0)
        )
        scale_coff = args.parse_float(
            "scale_coff",
            agent_cfg.get("sgp_scale_coff", 25)
        )
        if cl_method.endswith("_ref"):
            cl_mem = projection_memory_ref.ReferenceProjectionMemoryBank(
                device=device,
                threshold=threshold,
                method=cl_method.replace("_ref", ""),
                threshold_inc=threshold_inc,
                scale_coff=scale_coff,
            )
        else:
            compact_projection = agent_cfg.get("compact_projection", False)
            cl_mem = projection_memory.ProjectionMemoryBank(
                device=device,
                threshold=threshold,
                method=cl_method,
                threshold_inc=threshold_inc,
                scale_coff=scale_coff,
                compact_projection=compact_projection,
            )
        Logger.print("Projection memory: threshold={}, threshold_inc={}, scale_coff={}".format(
            threshold, threshold_inc, scale_coff))
    elif cl_method in EWC_METHODS:
        ewc_lambda = agent_cfg.get("ewc_lambda", 1000.0)
        ewc_online = agent_cfg.get("ewc_online", False)
        ewc_gamma = agent_cfg.get("ewc_gamma", 0.95)
        if cl_method == "ewc_ref":
            cl_mem = ewc_memory_ref.ReferenceEWCMemory(
                device=device,
                ewc_lambda=ewc_lambda,
                online=ewc_online,
                gamma=ewc_gamma,
            )
        else:
            cl_mem = ewc_memory.EWCMemory(
                device=device, ewc_lambda=ewc_lambda,
                online=ewc_online, gamma=ewc_gamma
            )
    else:
        assert False, "Unknown cl_method: {}".format(cl_method)

    Logger.print("CL method: {}".format(cl_method.upper()))

    # Load existing CL memory if resuming
    cl_memory_file = args.parse_string("cl_memory_file",
                                        args.parse_string("sgp_memory_file", ""))
    if cl_memory_file != "":
        cl_mem.load(cl_memory_file)
        if cl_method in PROJECTION_METHODS:
            Logger.print("Loaded CL memory from {} ({} tasks)".format(
                cl_memory_file, cl_mem.num_tasks))
        else:
            Logger.print("Loaded EWC memory from {} ({} tasks)".format(
                cl_memory_file, len(cl_mem.tasks)))

    # Build env ONCE using the first stage's config (IsaacGym PhysX Foundation
    # is a singleton — cannot be destroyed and recreated within a process).
    # Between stages, only the motion file is swapped via env._load_motions().
    Logger.print("Building environment (shared across all stages)...")
    env = env_builder.build_env_from_config(
        env_config=first_stage_config["env_config"],
        engine_config=first_stage_config["engine_config"],
        num_envs=first_stage_config["num_envs"],
        device=device,
        visualize=first_stage_config["visualize"],
        record_video=first_stage_config["record_video"]
    )

    # Performance matrices after each stage for multiple scalar metrics.
    perf_matrix_ep_len = []
    perf_matrix_return = []
    eval_rows_detailed = []
    stage_names = [s.get("name", "stage_{}".format(i))
                   for i, s in enumerate(stages[:end_stage + 1])]

    # Train each motion stage sequentially
    for stage_idx in range(start_stage, end_stage + 1):
        stage = stages[stage_idx]
        stage_config = cl_base.build_stage_config(stage_idx, stage, curriculum, args)
        task_id = stage_idx  # task_id = stage index
        stage_cl_method = stage_config["agent_config"].get("cl_method", cl_method)
        assert stage_cl_method == cl_method, \
            "Changing cl_method across stages is not supported: {} vs {}".format(
                stage_cl_method, cl_method
            )

        # Redirect stage output into the run directory
        stage_name = cl_base.sanitize_stage_name(stage_config["name"])
        stage_config["out_dir"] = os.path.join(
            curriculum_out_dir, "stage_{:02d}_{}".format(stage_idx, stage_name)
        )

        out_model_file, stage_record = train_cl_stage(
            env=env,
            stage_config=stage_config,
            task_id=task_id,
            device=device,
            in_model_file=in_model_file,
            cl_mem=cl_mem,
            curriculum_file=curriculum_config_file,
            cl_method=cl_method,
            cl_n_steps=args.parse_int("cl_n_steps", 300),
            cl_max_envs=args.parse_int("cl_max_envs", 256)
        )

        in_model_file = out_model_file
        stage_records.append(stage_record)

        # Evaluate all learned tasks after this stage
        eval_row = evaluate_after_stage(
            env, stages, stage_idx, out_model_file,
            device, curriculum, args
        )
        eval_rows_detailed.append(eval_row)
        perf_matrix_ep_len.append(extract_metric_row(eval_row, "mean_ep_len"))
        perf_matrix_return.append(extract_metric_row(eval_row, "mean_return"))
        stage_record["eval_after_stage"] = eval_row
        cl_base.save_yaml(stage_record, os.path.join(stage_config["out_dir"], "stage_metrics.yaml"))

        # Restore motion for next stage training
        if stage_idx < end_stage:
            next_motion = stages[stage_idx + 1].get("env_overrides", {}).get("motion_file", "")
            if next_motion:
                env._load_motions(next_motion)

    # Print full performance matrix and CL metrics
    print_performance_matrix(perf_matrix_ep_len, stage_names, metric_name="mean_ep_len")
    print_performance_matrix(perf_matrix_return, stage_names, metric_name="mean_return")

    cl_metrics = {
        "mean_ep_len": compute_cl_metrics(perf_matrix_ep_len, stage_names),
        "mean_return": compute_cl_metrics(perf_matrix_return, stage_names),
    }
    for metric_name, metric_values in cl_metrics.items():
        if metric_values:
            Logger.print("CL Metrics [{}]:".format(metric_name))
            Logger.print("  Average Performance (AP): {:.3f}".format(metric_values["average_performance"]))
            Logger.print("  Backward Transfer  (BWT): {:.3f}".format(metric_values["backward_transfer"]))
            Logger.print("  Forgetting         (FGT): {:.3f}".format(metric_values["forgetting"]))

    # Save performance matrix and metrics
    if mp_util.is_root_proc():
        matrix_data = {
            "performance_matrix_ep_len": perf_matrix_ep_len,
            "performance_matrix_return": perf_matrix_return,
            "evaluation_rows_detailed": eval_rows_detailed,
            "stage_names": stage_names,
            "cl_metrics": cl_metrics,
        }
        matrix_path = os.path.join(curriculum_out_dir, "cl_metrics.yaml")
        with open(matrix_path, "w") as f:
            yaml.safe_dump(matrix_data, f, sort_keys=False)
        Logger.print("CL metrics saved to {}".format(matrix_path))

    # Finalize
    cl_base.finalize_curriculum_outputs(curriculum_out_dir, stage_records,
                                        in_model_file)

    # Save final CL memory
    final_memory_path = os.path.join(curriculum_out_dir, "cl_memory_final.pt")
    cl_mem.save(final_memory_path)
    Logger.print("Final CL memory saved to {}".format(final_memory_path))
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
