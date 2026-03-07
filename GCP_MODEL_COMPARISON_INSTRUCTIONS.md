# GCP Model Comparison Runbook (Qwen + DeepSeek)

This runbook executes **one model at a time** over the exact same 100 prepared
threads (`batch_simulations_reconstructed/thread_001..thread_100`).

## 1. SSH and move into project

```bash
gcloud compute ssh <INSTANCE_NAME> --zone=<ZONE>
cd ~/Dissertation_Final
```

## 2. Pull latest code and verify thread set

```bash
git pull
python - <<'PY'
from pathlib import Path
inp = sorted([d.name for d in Path("batch_simulations_reconstructed").glob("thread_*") if d.is_dir()])
ref = sorted([d.name for d in Path("batch_output_new").glob("thread_*") if d.is_dir()])
print("input_threads=", len(inp))
print("baseline_threads=", len(ref))
print("same_set=", inp == ref)
PY
```

Expected:
- `input_threads= 100`
- `baseline_threads= 100`
- `same_set= True`

## 3. Set DeepSeek API key for this shell

```bash
export DEEPSEEK_API_KEY='<YOUR_KEY>'
```

## 4. Optional quick sanity check (no run)

```bash
python scripts/run_model_comparison_batch.py --model qwen --dry-run
python scripts/run_model_comparison_batch.py --model deepseek_r1 --dry-run
python scripts/run_model_comparison_batch.py --model deepseek_llm --dry-run
```

## 5. Run models sequentially with nohup

### 5.1 Qwen

```bash
./scripts/start_model_run_nohup.sh qwen
```

Monitor:

```bash
tail -f logs/model_runs/qwen_*.log
```

Wait for completion (`summary.json` exists):

```bash
ls -lh batch_output_qwen/summary.json
```

### 5.2 DeepSeek-R1 (local Ollama)

```bash
./scripts/start_model_run_nohup.sh deepseek_r1
```

Monitor:

```bash
tail -f logs/model_runs/deepseek_r1_*.log
```

Wait for completion:

```bash
ls -lh batch_output_deepseek_r1/summary.json
```

### 5.3 DeepSeek LLM API (`deepseek-chat`)

```bash
./scripts/start_model_run_nohup.sh deepseek_llm
```

Monitor:

```bash
tail -f logs/model_runs/deepseek_llm_*.log
```

Wait for completion:

```bash
ls -lh batch_output_deepseek_llm/summary.json
```

## 6. Validate each run against same real thread set

```bash
python scripts/validate_batch_old_params.py --variant new
```

The validator expects `batch_output_new` by default. To validate each new model
output, copy/symlink one output at a time to the expected path:

```bash
mv batch_output_new batch_output_new__backup
ln -s batch_output_qwen batch_output_new
python scripts/validate_batch_old_params.py --variant new
rm batch_output_new

ln -s batch_output_deepseek_r1 batch_output_new
python scripts/validate_batch_old_params.py --variant new
rm batch_output_new

ln -s batch_output_deepseek_llm batch_output_new
python scripts/validate_batch_old_params.py --variant new
rm batch_output_new

mv batch_output_new__backup batch_output_new
```

## 7. Download outputs back to local machine

Run from local machine:

```bash
gcloud compute scp --recurse <INSTANCE_NAME>:~/Dissertation_Final/batch_output_qwen ./ --zone=<ZONE>
gcloud compute scp --recurse <INSTANCE_NAME>:~/Dissertation_Final/batch_output_deepseek_r1 ./ --zone=<ZONE>
gcloud compute scp --recurse <INSTANCE_NAME>:~/Dissertation_Final/batch_output_deepseek_llm ./ --zone=<ZONE>
gcloud compute scp --recurse <INSTANCE_NAME>:~/Dissertation_Final/logs/model_runs ./logs --zone=<ZONE>
```

## Notes

- The `deepseek_llm` preset uses provider `deepseek` with model
  `deepseek-chat` and base URL `https://api.deepseek.com`.
- Keep one run active at a time to match your requested execution protocol.
