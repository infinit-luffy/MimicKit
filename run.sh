#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/mimickit"

CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-data/curriculums/cl_humanoid_example.yaml}"
RAND_SEED="${RAND_SEED:-42}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
CL_N_STEPS="${CL_N_STEPS:-300}"
CL_MAX_ENVS="${CL_MAX_ENVS:-256}"

COMMON_ARGS=(
  --curriculum_config "${CURRICULUM_CONFIG}"
  --rand_seed "${RAND_SEED}"
  --num_envs "${NUM_ENVS}"
  --max_samples "${MAX_SAMPLES}"
  --cl_n_steps "${CL_N_STEPS}"
  --cl_max_envs "${CL_MAX_ENVS}"
  --critic_optimizer Adam
  --disc_optimizer Adam
)

run_gpm() {
  python cl_run.py \
    "${COMMON_ARGS[@]}" \
    --cl_method gpm \
    --actor_optimizer Projection_Adam
}

run_sgp() {
  python cl_run.py \
    "${COMMON_ARGS[@]}" \
    --cl_method sgp \
    --actor_optimizer Projection_Adam
}

run_ewc() {
  python cl_run.py \
    "${COMMON_ARGS[@]}" \
    --cl_method ewc \
    --actor_optimizer Adam
}

run_gpm_ref() {
  python cl_run.py \
    "${COMMON_ARGS[@]}" \
    --cl_method gpm_ref \
    --actor_optimizer Projection_Adam
}

run_sgp_ref() {
  python cl_run.py \
    "${COMMON_ARGS[@]}" \
    --cl_method sgp_ref \
    --actor_optimizer Projection_Adam
}

run_ewc_ref() {
  python cl_run.py \
    "${COMMON_ARGS[@]}" \
    --cl_method ewc_ref \
    --actor_optimizer Adam
}

print_usage() {
  cat <<'EOF'
Usage:
  ./run.sh <algorithm>

Algorithms:
  gpm
  sgp
  ewc
  gpm_ref
  sgp_ref
  ewc_ref
  all

Environment overrides:
  CURRICULUM_CONFIG
  RAND_SEED
  NUM_ENVS
  MAX_SAMPLES
  CL_N_STEPS
  CL_MAX_ENVS

Examples:
  ./run.sh sgp
  RAND_SEED=7 MAX_SAMPLES=2000000 ./run.sh sgp_ref
  ./run.sh all
EOF
}

ALGO="${1:-}"

case "${ALGO}" in
  gpm)
    run_gpm
    ;;
  sgp)
    run_sgp
    ;;
  ewc)
    run_ewc
    ;;
  gpm_ref)
    run_gpm_ref
    ;;
  sgp_ref)
    run_sgp_ref
    ;;
  ewc_ref)
    run_ewc_ref
    ;;
  all)
    run_gpm
    run_sgp
    run_ewc
    run_gpm_ref
    run_sgp_ref
    run_ewc_ref
    ;;
  ""|-h|--help|help)
    print_usage
    ;;
  *)
    echo "Unknown algorithm: ${ALGO}" >&2
    print_usage
    exit 1
    ;;
esac
