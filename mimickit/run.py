"""
Unified entry point for MimicKit experiments.

Modes
-----
train     Single-motion PPO/AMP training (default).
test      Single-motion evaluation.
cl_train  Continual-learning sequential training via a curriculum file.
cl_eval   Evaluate a CL model across all curriculum motions.

Examples
--------
# Single-motion training
python run.py --mode train \
    --env_config data/envs/amp_humanoid_env.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --num_envs 4096 --visualize False

# Single-motion test
python run.py --mode test \
    --env_config data/envs/amp_humanoid_env.yaml \
    --agent_config data/agents/amp_humanoid_agent.yaml \
    --model_file output/model.pt

# CL training  (--algo overrides cl_method in the agent yaml)
python run.py --mode cl_train \
    --curriculum_config data/curriculums/cl_humanoid_example.yaml \
    --algo gpm

# CL training shorthand (curriculum_config presence implies cl_train)
python run.py \
    --curriculum_config data/curriculums/cl_humanoid_example.yaml \
    --algo gpm

# CL evaluation
python run.py --mode cl_eval \
    --curriculum_config data/curriculums/cl_humanoid_example.yaml \
    --model_file output/cl_humanoid/model.pt \
    --num_episodes 10
"""

import numpy as np
import os
import shutil
import sys
import time

import util.rsl_rl_util as rsl_rl_util
rsl_rl_util.configure_rsl_rl_path()

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize):
    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")
    record_video = args.parse_bool("video", False)

    env = env_builder.build_env(env_file, engine_file, num_envs, device, visualize=visualize, record_video=record_video)
    return env

def build_agent(args, env, device):
    agent_file = args.parse_string("agent_config")
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, out_dir, save_int_models, logger_type):
    agent.train_model(max_samples=max_samples, out_dir=out_dir,
                      save_int_models=save_int_models, logger_type=logger_type)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)

    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))
    return result

def save_config_files(args, out_dir):
    engine_file = args.parse_string("engine_config")
    if (engine_file != ""):
        copy_file_to_dir(engine_file, "engine_config.yaml", out_dir)

    env_file = args.parse_string("env_config")
    if (env_file != ""):
        copy_file_to_dir(env_file, "env_config.yaml", out_dir)

    agent_file = args.parse_string("agent_config")
    if (agent_file != ""):
        copy_file_to_dir(agent_file, "agent_config.yaml", out_dir)
    return

def create_output_dir(out_dir):
    if (mp_util.is_root_proc()):
        if (out_dir != "" and (not os.path.exists(out_dir))):
            os.makedirs(out_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)

    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def _infer_mode(args):
    """Infer mode: if curriculum_config is given, default to cl_train."""
    mode = args.parse_string("mode", "")
    if mode == "":
        if args.parse_string("curriculum_config") != "":
            mode = "cl_train"
        else:
            mode = "train"
    return mode

def run(rank, num_procs, device, master_port, args):
    mode = _infer_mode(args)

    if mode in ("cl_train", "cl_eval"):
        # Lazy import to avoid circular dependency (cl_run imports run as run_lib)
        if mode == "cl_train":
            import cl_run as cl_run_mod
            cl_run_mod.run(rank, num_procs, device, master_port, args,
                           cl_run_mod.resolve_rand_seed(args))
        else:
            import cl_eval as cl_eval_mod
            cl_eval_mod.run(rank, num_procs, device, master_port, args)
        return

    # ---- single-motion train / test ----
    num_envs = args.parse_int("num_envs", 1)
    visualize = args.parse_bool("visualize", True)
    logger_type = args.parse_string("logger", "txt")
    model_file = args.parse_string("model_file", "")

    out_dir = args.parse_string("out_dir", "output/")
    save_int_models = args.parse_bool("save_int_models", False)
    max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)

    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()
    create_output_dir(out_dir)

    env = build_env(args, num_envs, device, visualize)
    agent = build_agent(args, env, device)

    if (model_file != ""):
        agent.load(model_file)

    if (mode == "train"):
        save_config_files(args, out_dir)
        train(agent=agent, max_samples=max_samples, out_dir=out_dir,
              save_int_models=save_int_models, logger_type=logger_type)

    elif (mode == "test"):
        test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
        test(agent=agent, test_episodes=test_episodes)

    else:
        assert(False), "Unsupported mode: {}. Use train / test / cl_train / cl_eval".format(mode)

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

    mode = _infer_mode(args)

    processes = []
    for rank in range(1, num_workers):
        curr_device = devices[rank]
        if mode == "cl_train":
            import cl_run as cl_run_mod
            rand_seed = cl_run_mod.resolve_rand_seed(args)
            proc = torch.multiprocessing.Process(
                target=cl_run_mod.run,
                args=[rank, num_workers, curr_device, master_port, args, rand_seed])
        elif mode == "cl_eval":
            import cl_eval as cl_eval_mod
            proc = torch.multiprocessing.Process(
                target=cl_eval_mod.run,
                args=[rank, num_workers, curr_device, master_port, args])
        else:
            proc = torch.multiprocessing.Process(
                target=run,
                args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)

    root_device = devices[0]
    run(root_rank, num_workers, root_device, master_port, args)

    for proc in processes:
        proc.join()

    return

if __name__ == "__main__":
    main(sys.argv)
