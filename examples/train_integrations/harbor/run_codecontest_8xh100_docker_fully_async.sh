#!/usr/bin/env bash
set -ex

# ============================================================================
# SkyRL + Harbor: CodeContests fully async training on 8x H100 with Docker
#
# Usage:
#   ./run_codecontest_8xh100_docker_fully_async.sh smoke
#   ./run_codecontest_8xh100_docker_fully_async.sh pilot
#   ./run_codecontest_8xh100_docker_fully_async.sh full
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STAGE="${1:-smoke}"
if [ "$#" -gt 0 ]; then
  shift
fi

DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"
EVAL_DATA="['$DATA_DIR/OpenThoughts-TB-dev']"

RUN_NAME="${RUN_NAME_OVERRIDE:-codecontest-8xh100-docker-fully-async-${STAGE}}"
ENTRYPOINT_OVERRIDE="${ENTRYPOINT_OVERRIDE:-}"
PYTHONPATH_PREPEND="${PYTHONPATH_PREPEND:-}"
LOG_DIR="$HOME/tmp_logs/$RUN_NAME"
TENSORBOARD_DIR="$LOG_DIR/tensorboard"
RUN_ARTIFACT_ROOT_DEFAULT="$HOME"
if [ "$STAGE" = "full" ]; then
  RUN_ARTIFACT_ROOT_DEFAULT="/scratch/$USER"
fi
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-$RUN_ARTIFACT_ROOT_DEFAULT}"
RUN_ARTIFACT_DIR="$RUN_ARTIFACT_ROOT/$RUN_NAME"
TRIALS_DIR="$RUN_ARTIFACT_DIR/trials_run"
CKPTS_DIR="$RUN_ARTIFACT_DIR/ckpts"
EXPORTS_DIR="$RUN_ARTIFACT_DIR/exports"
MONITORING_DIR="$LOG_DIR/monitoring"
ASYNC_TRACE_PATH="$MONITORING_DIR/async_trace.jsonl"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/skyrl_runtime}"
TMP_ROOT="$SCRATCH_ROOT/ray_tmp"
UV_CACHE_DIR="$SCRATCH_ROOT/uv-codex"
TORCHINDUCTOR_CACHE_DIR="$SCRATCH_ROOT/torchinductor"
TRITON_CACHE_DIR="$SCRATCH_ROOT/triton"
HF_ROOT="/data/zy/models"
HF_HUB_DIR="$HF_ROOT/hub"
HF_XET_DIR="$HF_ROOT/xet"

mkdir -p \
  "$TRIALS_DIR" \
  "$CKPTS_DIR" \
  "$EXPORTS_DIR" \
  "$LOG_DIR" \
  "$MONITORING_DIR" \
  "$SCRATCH_ROOT" \
  "$TMP_ROOT" \
  "$UV_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$HF_HUB_DIR" \
  "$HF_XET_DIR"

export TMPDIR="$TMP_ROOT"
export RAY_TMPDIR="$TMP_ROOT"
export UV_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
export TENSORBOARD_DIR
export HF_HOME="$HF_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_DIR"
export HF_HUB_CACHE="$HF_HUB_DIR"
export TRANSFORMERS_CACHE="$HF_HUB_DIR"
export HF_XET_CACHE="$HF_XET_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SKYRL_ASYNC_TRACE_PATH="$ASYNC_TRACE_PATH"

MODEL_PATH="Qwen/Qwen3-8B"
MODEL_NAME="Qwen3-8B"
MAX_MODEL_LEN=32768
TRAIN_MAX_SEQ_LEN=6144
CHAT_TEMPLATE_PATH="$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN"
  exit 1
fi

PYTHONPATH_PARTS=("$REPO_ROOT")
if [ -n "$PYTHONPATH_PREPEND" ]; then
  PYTHONPATH_PARTS=("$PYTHONPATH_PREPEND" "${PYTHONPATH_PARTS[@]}")
fi
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH_PARTS+=("$PYTHONPATH")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"

TRAIN_GPUS=4
ROLLOUT_ENGINES=4

LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=true
KL_LOSS_COEF=0.001
TIS_TYPE="${TIS_TYPE:-token}"
TIS_IMP_RATIO_CAP="${TIS_IMP_RATIO_CAP:-2.0}"
APPLY_OVERLONG_FILTERING=true
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"
MONITOR_PID=""
FLASH_ATTN=false
USE_SAMPLE_PACKING=false
AGENT_TEMPERATURE=0.3
ENABLE_THINKING=false
INCLUDE_REASONING=false

cleanup_monitor() {
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}

trap cleanup_monitor EXIT INT TERM

case "$STAGE" in
  smoke)
    ENTRYPOINT="examples.train_integrations.harbor.entrypoints.main_harbor_fully_async"
    MAX_TRAIN_TASKS=8
    MAX_EVAL_TASKS=null
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE=2
    MINI_BATCH_SIZE=2
    EPOCHS=1
    EVAL_BEFORE_TRAIN=false
    EVAL_INTERVAL=0
    TRAJ_PER_SEC=1
    MAX_CONCURRENCY=4
    LOGGER=console
    COLLECT_MEMORY_METRICS=false
    AUTO_START_MONITOR=false
    MAX_TURNS=8
    TIMEOUT_SEC=900
    MAX_STALENESS_STEPS=1
    NUM_PARALLEL_GENERATION_WORKERS=2
    ;;
  pilot)
    ENTRYPOINT="examples.train_integrations.harbor.entrypoints.main_harbor_fully_async"
    MAX_TRAIN_TASKS=64
    MAX_EVAL_TASKS=10
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE=4
    MINI_BATCH_SIZE=4
    EPOCHS=1
    EVAL_BEFORE_TRAIN=false
    EVAL_INTERVAL=20
    TRAJ_PER_SEC=1
    MAX_CONCURRENCY=8
    LOGGER=console
    COLLECT_MEMORY_METRICS=false
    AUTO_START_MONITOR=false
    MAX_TURNS=10
    TIMEOUT_SEC=900
    MAX_STALENESS_STEPS=1
    NUM_PARALLEL_GENERATION_WORKERS=4
    ;;
  full)
    ENTRYPOINT="examples.train_integrations.harbor.entrypoints.main_harbor_fully_async"
    MAX_TRAIN_TASKS=null
    MAX_EVAL_TASKS=20
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE=4
    MINI_BATCH_SIZE=4
    EPOCHS=1
    EVAL_BEFORE_TRAIN=false
    EVAL_INTERVAL=50
    TRAJ_PER_SEC=2
    MAX_CONCURRENCY=8
    LOGGER="['tensorboard','console']"
    COLLECT_MEMORY_METRICS=true
    AUTO_START_MONITOR=true
    MAX_TURNS=10
    TIMEOUT_SEC=900
    MAX_STALENESS_STEPS=1
    NUM_PARALLEL_GENERATION_WORKERS=4
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|full}"
    exit 1
    ;;
esac

if [ -n "$ENTRYPOINT_OVERRIDE" ]; then
  ENTRYPOINT="$ENTRYPOINT_OVERRIDE"
fi

THUNDER_AGENT_MODE="${THUNDER_AGENT_MODE:-tr}"
THUNDER_AGENT_PROFILE_ENABLED="${THUNDER_AGENT_PROFILE_ENABLED:-true}"
THUNDER_AGENT_METRICS_ENABLED="${THUNDER_AGENT_METRICS_ENABLED:-true}"
THUNDER_AGENT_EXTRA_ARGS=()
if [[ "$ENTRYPOINT" == *"thunder_agent"* ]]; then
  THUNDER_AGENT_EXTRA_ARGS=(
    generator.inference_engine.thunder_agent_mode="$THUNDER_AGENT_MODE"
    generator.inference_engine.thunder_agent_profile_enabled="$THUNDER_AGENT_PROFILE_ENABLED"
    generator.inference_engine.thunder_agent_metrics_enabled="$THUNDER_AGENT_METRICS_ENABLED"
  )
fi

if [ "$STAGE" = full ] && [ -n "${WANDB_API_KEY:-}" ]; then
  LOGGER="['wandb','tensorboard','console']"
fi

if [ "$AUTO_START_MONITOR" = true ]; then
  bash "$SCRIPT_DIR/monitor_stage3_resources.sh" "$RUN_NAME" "$MONITOR_INTERVAL_SEC" "$MONITORING_DIR" &
  MONITOR_PID=$!
  echo "Started fully-async monitor pid=$MONITOR_PID output_dir=$MONITORING_DIR tensorboard_dir=$TENSORBOARD_DIR"
fi

"$PYTHON_BIN" -m "$ENTRYPOINT" \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  max_train_tasks="$MAX_TRAIN_TASKS" \
  max_eval_tasks="$MAX_EVAL_TASKS" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.served_model_name="$MODEL_NAME" \
  harbor_trial_config.trials_dir="$TRIALS_DIR" \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.strategy=fsdp2 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.off_policy_correction.tis_ratio_type="$TIS_TYPE" \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high="$TIS_IMP_RATIO_CAP" \
  trainer.algorithm.loss_reduction="$LOSS_REDUCTION" \
  trainer.algorithm.grpo_norm_by_std="$GRPO_NORM_BY_STD" \
  trainer.algorithm.use_kl_loss="$USE_KL_LOSS" \
  trainer.algorithm.kl_loss_coef="$KL_LOSS_COEF" \
  trainer.algorithm.max_seq_len="$TRAIN_MAX_SEQ_LEN" \
  trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$TRAIN_GPUS" \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node="$TRAIN_GPUS" \
  trainer.critic.model.path=null \
  trainer.epochs="$EPOCHS" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train="$EVAL_BEFORE_TRAIN" \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.flash_attn="$FLASH_ATTN" \
  trainer.use_sample_packing="$USE_SAMPLE_PACKING" \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES" \
  generator.apply_overlong_filtering="$APPLY_OVERLONG_FILTERING" \
  generator.sampling_params.temperature="$AGENT_TEMPERATURE" \
  generator.sampling_params.logprobs=1 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.logprobs=1 \
  generator.inference_engine.num_engines="$ROLLOUT_ENGINES" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.inference_engine.engine_init_kwargs.chat_template="$CHAT_TEMPLATE_PATH" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second="$TRAJ_PER_SEC" \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  harbor_trial_config.environment.type=docker \
  harbor_trial_config.environment.override_cpus=2 \
  harbor_trial_config.environment.override_memory_mb=4096 \
  harbor_trial_config.environment.override_storage_mb=4096 \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=null \
  harbor_trial_config.agent.override_timeout_sec="$TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.enable_summarize=false \
  harbor_trial_config.agent.kwargs.store_all_messages=true \
  harbor_trial_config.agent.kwargs.temperature="$AGENT_TEMPERATURE" \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning="$INCLUDE_REASONING" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN" \
  trainer.logger="$LOGGER" \
  trainer.collect_memory_metrics="$COLLECT_MEMORY_METRICS" \
  trainer.collect_memory_metrics_interval=1 \
  trainer.project_name=harbor \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  "${THUNDER_AGENT_EXTRA_ARGS[@]}" \
  "$@"
