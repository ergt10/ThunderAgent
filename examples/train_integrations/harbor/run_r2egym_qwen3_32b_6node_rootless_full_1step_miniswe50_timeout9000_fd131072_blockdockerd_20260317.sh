#!/usr/bin/env bash
set -euo pipefail

cd /home/hkang/zthunder_agent/SkyRL
export PATH=/home/hkang/zthunder_agent/SkyRL/.venv/bin:$HOME/.local/bin:$PATH

export TRAIN_DATA="['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']"
export EVAL_DATA="$TRAIN_DATA"
export RAY_ADDRESS="172.21.44.54:6381"
export RAY_HEAD_IP="172.21.44.54"
export ROLLOUT_HOST_IP="172.21.44.94"
export ROLLOUT_SERVER_PORTS_CSV="18000,18001,18002,18003"
export ROLLOUT_ENGINES=4
export ROLLOUT_TP_SIZE=2
export DOCKER_MODE=rootless
export RUN_NAME_OVERRIDE="r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051"
export LOG_DIR_OVERRIDE="/home/hkang/zthunder_agent/tmp_logs/r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-20260317_220051"
export RUN_ARTIFACT_ROOT="/home/hkang/zthunder_agent"
export SCRATCH_ROOT="/tmp/hkang/r2e1s-0317220051b-train-runtime"
export XDG_RUNTIME_DIR="/tmp/xdg-r2e1s-0317220051b"
export DOCKER_HOST="unix:///tmp/xdg-r2e1s-0317220051b/docker.sock"
export DOCKER_PIDFILE="/tmp/xdg-r2e1s-0317220051b/docker.pid"
export DOCKER_EXEC_ROOT="/tmp/hkang/r2e1s-0317220051b-rootless-exec"
export DOCKER_DATA_ROOT="/scratch/triton_cache/hkang/r2e1s-0317220051b-rootless-data"
export HEAD_NOFILE_SOFT=131072
export ROLLOUT_METRICS_ENDPOINT_SPECS="rollout_a.log=http://172.21.44.94:18000;rollout_b.log=http://172.21.44.94:18001;rollout_c.log=http://172.21.44.94:18002;rollout_d.log=http://172.21.44.94:18003"
export RUN_PREFLIGHT_CHECKS=false

exec stdbuf -oL -eL \
  bash examples/train_integrations/harbor/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh full \
    max_train_tasks=64 \
    trainer.resume_mode=none \
    harbor_trial_config.agent.name=mini-swe-agent \
    harbor_trial_config.agent.kwargs.max_turns=50 \
    harbor_trial_config.agent.override_timeout_sec=9000
