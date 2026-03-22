# Continual Learning / Sequential Mimic Training

This repository now includes a stage-wise continual training entrypoint:

```bash
python mimickit/continual_run.py --arg_file args/deepmimic_humanoid_continual_args.txt
```

The default example curriculum is:

- [`data/curricula/deepmimic_humanoid_continual.yaml`](../data/curricula/deepmimic_humanoid_continual.yaml)

The default argument file is:

- [`args/deepmimic_humanoid_continual_args.txt`](../args/deepmimic_humanoid_continual_args.txt)


## What It Does

This is currently a **sequential training pipeline**, not an algorithm-level continual learning method yet.

It will:

1. Train stage 0.
2. Save the checkpoint to that stage's output directory.
3. Load the previous stage checkpoint into the next stage automatically.
4. Continue training on the next motion / dataset.

So this is the right scaffold for:

- curriculum learning
- stage-wise motion expansion
- continual learning experiments after you modify the RL algorithm


## Default Run Command

Run the full example curriculum:

```bash
python mimickit/continual_run.py \
  --arg_file args/deepmimic_humanoid_continual_args.txt
```

This uses:

- engine: `data/engines/isaac_gym_engine.yaml`
- curriculum: `data/curricula/deepmimic_humanoid_continual.yaml`
- output dir: `output/deepmimic_humanoid_continual`


## Resume From A Specific Stage

Run only stages 1 to 2:

```bash
python mimickit/continual_run.py \
  --arg_file args/deepmimic_humanoid_continual_args.txt \
  --start_stage 1 \
  --end_stage 2
```

Notes:

- `start_stage` and `end_stage` are zero-based.
- By default, stage 1 will initialize from the previous stage model if you provide `--model_file` or if the previous stage was just trained in the same run.


## Start From An Existing Checkpoint

If you already have a pretrained model and want to begin continual training from it:

```bash
python mimickit/continual_run.py \
  --arg_file args/deepmimic_humanoid_continual_args.txt \
  --model_file /absolute/path/to/model.pt
```


## Multi-GPU Run

Use the same multi-process interface as the original training script:

```bash
python mimickit/continual_run.py \
  --arg_file args/deepmimic_humanoid_continual_args.txt \
  --devices cuda:0 cuda:1
```


## Using A Different Local rsl_rl Path

The entry scripts automatically prepend the local repo:

- `third_party/rsl_rl`

If you want to override it:

```bash
RSL_RL_ROOT=/absolute/path/to/rsl_rl \
python mimickit/continual_run.py \
  --arg_file args/deepmimic_humanoid_continual_args.txt
```


## Output Layout

For each stage, the script writes:

- `env_config.yaml`
- `agent_config.yaml`
- `engine_config.yaml`
- `stage_info.yaml`
- `model.pt`
- `log.txt` or TensorBoard files depending on logger

Example output tree:

```text
output/deepmimic_humanoid_continual/
  stage_00_walk/
  stage_01_run/
  stage_02_locomotion_mix/
  curriculum_summary.yaml
  model.pt
```


## Current Limitation

At this point, the algorithm itself is still the original MimicKit training algorithm.

So this setup does **not** yet include:

- replay buffers across stages
- EWC / LwF / distillation losses
- policy regularization against older tasks
- task-ID conditioning

Those should be added in the RL algorithm implementation next.
