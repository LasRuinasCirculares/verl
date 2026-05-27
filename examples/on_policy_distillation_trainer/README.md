# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                                                               | Teachers | Modality  | Infer | Train    | Platform |
|----------------------------------------------------------------------|----------|-----------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`                                               | single   | text      | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`                                           | single   | text      | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`                                            | single   | VL        | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`                                          | multi    | text + VL | vLLM  | FSDP     | NVIDIA   |
| `run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_sbatch.sh`       | single   | text      | vLLM  | FSDP     | NVIDIA + Slurm |
| `run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_4gpu_sbatch.sh`  | single   | text      | vLLM  | FSDP     | NVIDIA + Slurm |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

Submit the DeepSeek-R1-Distill-Qwen 1.5B <- 7B Slurm recipe from the `verl-main`
root with:

```bash
sbatch examples/on_policy_distillation_trainer/run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_sbatch.sh
```

That script requests 8 GPUs and defaults to a 4-GPU trainer/rollout resource
pool plus a separate 4-GPU teacher resource pool. Override
`TRAINER_NGPUS_PER_NODE`, `TEACHER_NGPUS_PER_NODE`, and `TOTAL_GPUS`
consistently if you change the split.

For a 4-GPU node, submit:

```bash
sbatch examples/on_policy_distillation_trainer/run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_4gpu_sbatch.sh
```

The 4-GPU wrapper defaults to a 2-GPU trainer/rollout resource pool plus a
separate 2-GPU teacher resource pool and reuses the same training recipe.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`
