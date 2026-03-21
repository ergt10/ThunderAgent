# Qwen3-32B 10-Step Full Run Capture

This wrapper keeps the same `full` training and rollout parameters as
`run_codecontest_qwen3_32b_6node_rootless_fully_async.sh`, but changes the
run-length and artifact policy so it is safe for a short validation run:

- `max_train_tasks = policy_mini_batch_size * 10`
- checkpoints go to `/home` via `RUN_ARTIFACT_ROOT`
- `trainer.ckpt_interval=10`
- `trainer.max_ckpts_to_keep=1`
- `trainer.save_final_checkpoint_at_end=false`
- `trainer.hf_save_interval=-1`

It also:

- runs rollout CUDA runtime smoke on the rollout node
- starts Ray on `003 + 012-015`
- starts rollout servers on `008`
- runs the reusable full validation suite
- starts one GPU monitor per trainer node
- produces a post-run summary with:
  - trainer GPU memory timeline
  - rollout KV-cache timeline
  - per-step training timing breakdown

Entry point:

```bash
JOB_ID=<slurm-job-id> \
bash /home/hkang/zthunder_agent/SkyRL/examples/train_integrations/harbor/run_qwen3_32b_full_10step_capture.sh
```
