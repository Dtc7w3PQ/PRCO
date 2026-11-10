# PRCO training

PRCO-7B Observer/Solver training with Qwen2.5-VL-7B-Instruct, configured for **200 steps**. Based on EasyR1 and veRL.

## Setup

Use Python 3.12 and compatible CUDA, PyTorch and vLLM versions.

```bash
cd training
pip install -e .
```

## Data

Example layout under `DATA_ROOT`:

```text
data/
  train.jsonl
  val.jsonl
  train_images/
  val_images/
```

Each JSONL record contains `problem`, `answer` and `images`. Image paths are relative to the corresponding image directory.

```json
{"problem": "What is the value of x?", "answer": "4", "images": ["example.png"]}
```

## Train

Start a judge service on separate GPUs:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 9200
```

Then launch training, replacing the example host and port with your judge endpoint:

```bash
DATA_ROOT=./data VLLM_JUDGE_IP=127.0.0.1 VLLM_JUDGE_PORTS=9200 \
bash scripts/train_prco_7b.sh
```

Use `--dry-run` to preview the command. `TRAIN_JSONL`, `VAL_JSONL`, `TRAIN_IMAGE_DIR`, `VAL_IMAGE_DIR`, `MODEL_PATH` and `OUTPUT_DIR` can be overridden through environment variables. Relative paths use the directory where you launch the script.

Training settings are in `configs/prco_7b.yaml`. Logs and checkpoints default to `./outputs/prco_7b/`.

## Export

```bash
python scripts/export_checkpoint.py --local_dir outputs/prco_7b/checkpoints/global_step_200/actor
```

The model is saved in the checkpoint's `huggingface/` subdirectory. Default checkpoints contain model weights only.
