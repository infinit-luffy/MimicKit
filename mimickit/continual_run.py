import copy
import numpy as np
import os
import re
import shutil
import sys
import yaml

import util.rsl_rl_util as rsl_rl_util
rsl_rl_util.configure_rsl_rl_path()

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import run as run_lib
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util

import torch


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args


def load_yaml(file):
    with open(file, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def merge_dicts(base_dict, override_dict):
    if (base_dict is None):
        result = dict()
    else:
        result = copy.deepcopy(base_dict)

    if (override_dict is None):
        return result

    for key, val in override_dict.items():
        if (isinstance(val, dict) and isinstance(result.get(key), dict)):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = copy.deepcopy(val)

    return result


def sanitize_stage_name(stage_name):
    safe_name = re.sub(r"[^0-9a-zA-Z]+", "_", stage_name).strip("_")
    if (safe_name == ""):
        safe_name = "stage"
    return safe_name.lower()


def create_output_dir(out_dir):
    if (mp_util.is_root_proc()):
        os.makedirs(out_dir, exist_ok=True)
    return


def save_yaml(data, out_file):
    if (mp_util.is_root_proc()):
        with open(out_file, "w") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
    return


def load_curriculum(curriculum_file):
    curriculum = load_yaml(curriculum_file)
    assert(curriculum is not None), "Empty curriculum config: {}".format(curriculum_file)
    stages = curriculum.get("stages", [])
    assert(len(stages) > 0), "Curriculum must define at least one stage"
    return curriculum


def build_stage_config(stage_idx, stage, curriculum, args):
    stage_name = stage.get("name", "stage_{:02d}".format(stage_idx))

    env_file = stage.get("env_config", curriculum.get("env_config", args.parse_string("env_config")))
    agent_file = stage.get("agent_config", curriculum.get("agent_config", args.parse_string("agent_config")))
    engine_file = stage.get("engine_config", curriculum.get("engine_config", args.parse_string("engine_config")))

    assert(env_file != ""), "Missing env_config for stage {}".format(stage_name)
    assert(agent_file != ""), "Missing agent_config for stage {}".format(stage_name)

    env_config = env_builder.load_config(env_file)
    agent_config = agent_builder.load_agent_file(agent_file)
    engine_config = env_builder.load_config(engine_file)

    env_overrides = merge_dicts(curriculum.get("env_overrides", None), stage.get("env_overrides", None))
    agent_overrides = merge_dicts(curriculum.get("agent_overrides", None), stage.get("agent_overrides", None))
    engine_overrides = merge_dicts(curriculum.get("engine_overrides", None), stage.get("engine_overrides", None))

    env_config = merge_dicts(env_config, env_overrides)
    agent_config = merge_dicts(agent_config, agent_overrides)

    if (engine_overrides is not None and len(engine_overrides) > 0):
        engine_config = env_builder.override_engine_config(engine_overrides, engine_config)

    max_samples = stage.get("max_samples", curriculum.get("max_samples", args.parse_int("max_samples", np.iinfo(np.int64).max)))
    test_episodes = stage.get("test_episodes", curriculum.get("test_episodes", None))
    if (test_episodes is not None):
        agent_config["test_episodes"] = test_episodes

    logger_type = stage.get("logger", curriculum.get("logger", args.parse_string("logger", "txt")))
    save_int_models = stage.get("save_int_models", curriculum.get("save_int_models", args.parse_bool("save_int_models", False)))
    visualize = stage.get("visualize", curriculum.get("visualize", args.parse_bool("visualize", True)))
    record_video = stage.get("video", curriculum.get("video", args.parse_bool("video", False)))
    num_envs = stage.get("num_envs", curriculum.get("num_envs", args.parse_int("num_envs", 1)))

    base_out_dir = args.parse_string("out_dir", "output/")
    curriculum_out_dir = curriculum.get("out_dir", base_out_dir)
    default_stage_dir = os.path.join(curriculum_out_dir, "stage_{:02d}_{}".format(stage_idx, sanitize_stage_name(stage_name)))
    out_dir = stage.get("out_dir", default_stage_dir)

    stage_config = {
        "name": stage_name,
        "index": stage_idx,
        "env_file": env_file,
        "agent_file": agent_file,
        "engine_file": engine_file,
        "env_config": env_config,
        "agent_config": agent_config,
        "engine_config": engine_config,
        "max_samples": max_samples,
        "logger_type": logger_type,
        "save_int_models": save_int_models,
        "visualize": visualize,
        "record_video": record_video,
        "num_envs": num_envs,
        "out_dir": out_dir,
        "model_file": stage.get("model_file", ""),
        "reset_model": stage.get("reset_model", False)
    }
    return stage_config


def calc_env_signature(env):
    obs_space = env.get_obs_space()
    action_space = env.get_action_space()

    action_shape = getattr(action_space, "shape", None)
    signature = {
        "obs_shape": list(obs_space.shape),
        "action_type": type(action_space).__name__,
        "action_shape": list(action_shape) if action_shape is not None else None
    }
    return signature


def save_stage_files(stage_config, curriculum_file, in_model_file):
    out_dir = stage_config["out_dir"]
    create_output_dir(out_dir)

    env_out_file = os.path.join(out_dir, "env_config.yaml")
    agent_out_file = os.path.join(out_dir, "agent_config.yaml")
    engine_out_file = os.path.join(out_dir, "engine_config.yaml")
    stage_out_file = os.path.join(out_dir, "stage_info.yaml")

    save_yaml(stage_config["env_config"], env_out_file)
    save_yaml(stage_config["agent_config"], agent_out_file)
    save_yaml(stage_config["engine_config"], engine_out_file)

    stage_info = {
        "name": stage_config["name"],
        "index": stage_config["index"],
        "curriculum_file": curriculum_file,
        "source_env_config": stage_config["env_file"],
        "source_agent_config": stage_config["agent_file"],
        "source_engine_config": stage_config["engine_file"],
        "max_samples": int(stage_config["max_samples"]),
        "logger": stage_config["logger_type"],
        "save_int_models": bool(stage_config["save_int_models"]),
        "visualize": bool(stage_config["visualize"]),
        "video": bool(stage_config["record_video"]),
        "num_envs": int(stage_config["num_envs"]),
        "init_model_file": in_model_file
    }
    save_yaml(stage_info, stage_out_file)
    return


def finalize_curriculum_outputs(curriculum_out_dir, stage_records, last_model_file):
    if (not mp_util.is_root_proc()):
        return

    summary = {"stages": stage_records}
    save_yaml(summary, os.path.join(curriculum_out_dir, "curriculum_summary.yaml"))

    if (last_model_file != "" and os.path.exists(last_model_file)):
        shutil.copy(last_model_file, os.path.join(curriculum_out_dir, "model.pt"))
    return


def train_stage(stage_config, curriculum_file, device, in_model_file, prev_signature):
    stage_name = stage_config["name"]
    Logger.print("")
    Logger.print("=== Continual Stage {:02d}: {} ===".format(stage_config["index"], stage_name))

    save_stage_files(stage_config, curriculum_file=curriculum_file, in_model_file=in_model_file)

    env = env_builder.build_env_from_config(env_config=stage_config["env_config"],
                                            engine_config=stage_config["engine_config"],
                                            num_envs=stage_config["num_envs"],
                                            device=device,
                                            visualize=stage_config["visualize"],
                                            record_video=stage_config["record_video"])
    agent = agent_builder.build_agent_from_config(stage_config["agent_config"], env, device)

    curr_signature = calc_env_signature(env)
    if (prev_signature is not None):
        assert(curr_signature == prev_signature), \
            "Stage {} changed observation/action spaces: {} vs {}".format(stage_name, curr_signature, prev_signature)

    model_file = stage_config["model_file"]
    if (model_file == "" and (not stage_config["reset_model"])):
        model_file = in_model_file

    if (model_file != ""):
        agent.load(model_file)

    agent.train_model(max_samples=stage_config["max_samples"],
                      out_dir=stage_config["out_dir"],
                      save_int_models=stage_config["save_int_models"],
                      logger_type=stage_config["logger_type"])

    out_model_file = os.path.join(stage_config["out_dir"], "model.pt")
    stage_record = {
        "index": stage_config["index"],
        "name": stage_name,
        "out_dir": stage_config["out_dir"],
        "model_file": out_model_file,
        "init_model_file": model_file,
        "max_samples": int(stage_config["max_samples"])
    }
    return out_model_file, curr_signature, stage_record


def run(rank, num_procs, device, master_port, args):
    mode = args.parse_string("mode", "train")
    assert(mode == "train"), "continual_run.py currently supports train mode only"

    curriculum_config_file = args.parse_string("curriculum_config")
    assert(curriculum_config_file != ""), "Missing --curriculum_config"

    curriculum = load_curriculum(curriculum_config_file)
    stages = curriculum["stages"]

    start_stage = args.parse_int("start_stage", 0)
    end_stage = args.parse_int("end_stage", len(stages) - 1)
    assert(0 <= start_stage < len(stages))
    assert(start_stage <= end_stage < len(stages))

    curriculum_out_dir = curriculum.get("out_dir", args.parse_string("out_dir", "output/"))

    mp_util.init(rank, num_procs, device, master_port)
    run_lib.set_rand_seed(args)
    run_lib.set_np_formatting()
    create_output_dir(curriculum_out_dir)

    in_model_file = args.parse_string("model_file", curriculum.get("model_file", ""))
    prev_signature = None
    stage_records = []

    for stage_idx in range(start_stage, end_stage + 1):
        stage = stages[stage_idx]
        stage_config = build_stage_config(stage_idx, stage, curriculum, args)
        out_model_file, prev_signature, stage_record = train_stage(stage_config, curriculum_config_file, device, in_model_file, prev_signature)
        in_model_file = out_model_file
        stage_records.append(stage_record)

    finalize_curriculum_outputs(curriculum_out_dir, stage_records, in_model_file)
    return


def main(argv):
    root_rank = 0
    args = load_args(argv)
    master_port = args.parse_int("master_port", None)
    devices = args.parse_strings("devices", ["cuda:0"])

    num_workers = len(devices)
    assert(num_workers > 0)

    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for rank in range(1, num_workers):
        curr_device = devices[rank]
        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)

    root_device = devices[0]
    run(root_rank, num_workers, root_device, master_port, args)

    for proc in processes:
        proc.join()

    return


if __name__ == "__main__":
    main(sys.argv)
