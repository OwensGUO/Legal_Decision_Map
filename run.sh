#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
INFER_PID=""
INFER_PGID=""

usage() {
  cat <<'EOF'
Usage: bash run.sh [--dry-run] [--help]

Run the complete legal decision landscape pipeline inside an already activated
Conda environment. The script never creates or switches Conda environments.

Modes (set with MODE=main|smoke|matrix):
  main    Build both datasets, generate counterfactuals, train B3 and M with
          seed 42, export predictions, and run paired evaluation. This is default.
  smoke   Run the same pipeline with small limits and one training step.
  matrix  Run B0-B5, M, and A1-A6 with seeds 42, 2026, and 3407 on both datasets.

Common overrides:
  CAIL_ROOT=/data/.../CAIL2018     CMDL_ROOT=/data/.../CMDL
  QWEN35_PATH=/data/.../Qwen3.5-9B
  QWEN36_PATH=/data/.../Qwen3.6-27B
  ROBERTA_PATH=/data/.../roberta     LAWFORMER_PATH=/data/.../Lawformer
  GPU_IDS=4,5,6,7                 OUTPUT_ROOT=/data/.../outputs
  NUM_PROCESSES=4                 EXPERIMENTS="M" SEEDS="42"
  INSTALL_DEPS=1                  Install the pinned dependencies first
  MOCK_GENERATOR=1                Use the deterministic mock generator
  INFER_MANAGED=0                 Use an already-running vLLM endpoint
  INFER_HOST=127.0.0.1            INFER_PORT=30000 INFER_TIMEOUT=1800
  INFER_MODEL_NAME=Qwen3.6-27B    Served model name for probes and requests
  INFER_GPU_MEMORY=0.90           vLLM --gpu-memory-utilization
  VLLM_ENFORCE_EAGER=1           Avoid CUDA graphs on the driver-535 smoke path
  VLLM_DISABLE_P2P=1             Set NCCL_P2P_DISABLE=1 for RTX 4090
  DATA_LIMIT=N CF_LIMIT=N EVAL_LIMIT=N MAX_STEPS=N
  BOOTSTRAP_ITERATIONS=2000       Group-clustered bootstrap resamples
  CHECKPOINT_EVERY=100             Save resumable training state every N updates
  INSTALL_FLA=1                  Install and probe optional FLA/Triton kernels

Examples:
  conda activate legal-landscape
  MODE=smoke bash run.sh
  bash run.sh
  MODE=matrix bash run.sh
EOF
}

for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$argument" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MODE="${MODE:-main}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
CAIL_ROOT="${CAIL_ROOT:-/data/chenguo/datasets/CAIL2018}"
CMDL_ROOT="${CMDL_ROOT:-/data/chenguo/datasets/CMDL}"
QWEN35_PATH="${QWEN35_PATH:-/data/chenguo/Qwen3.5-9B}"
QWEN36_PATH="${QWEN36_PATH:-/data/chenguo/Qwen3.6-27B}"
ROBERTA_PATH="${ROBERTA_PATH:-/data/chenguo/models/RoBERTa}"
LAWFORMER_PATH="${LAWFORMER_PATH:-/data/chenguo/models/Lawformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}"
INFER_HOST="${INFER_HOST:-127.0.0.1}"
INFER_PORT="${INFER_PORT:-30000}"
INFER_TIMEOUT="${INFER_TIMEOUT:-1800}"
INFER_MANAGED="${INFER_MANAGED:-1}"
INFER_GPU_MEMORY="${INFER_GPU_MEMORY:-0.90}"
INFER_MODEL_NAME="${INFER_MODEL_NAME:-Qwen3.6-27B}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_DISABLE_P2P="${VLLM_DISABLE_P2P:-1}"
MOCK_GENERATOR="${MOCK_GENERATOR:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
INSTALL_FLA="${INSTALL_FLA:-0}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-}"

case "$MODE" in
  smoke)
    DATA_LIMIT="${DATA_LIMIT:-16}"
    CF_LIMIT="${CF_LIMIT:-12}"
    EVAL_LIMIT="${EVAL_LIMIT:-16}"
    MAX_STEPS="${MAX_STEPS:-1}"
    BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-50}"
    EXPERIMENTS="${EXPERIMENTS:-M}"
    SEEDS="${SEEDS:-42}"
    ;;
  main)
    DATA_LIMIT="${DATA_LIMIT:-1000000}"
    CF_LIMIT="${CF_LIMIT:-1000000}"
    EVAL_LIMIT="${EVAL_LIMIT:-1000000}"
    MAX_STEPS="${MAX_STEPS:-1000}"
    BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
    EXPERIMENTS="${EXPERIMENTS:-B3 M}"
    SEEDS="${SEEDS:-42}"
    ;;
  matrix)
    DATA_LIMIT="${DATA_LIMIT:-1000000}"
    CF_LIMIT="${CF_LIMIT:-1000000}"
    EVAL_LIMIT="${EVAL_LIMIT:-1000000}"
    MAX_STEPS="${MAX_STEPS:-1000}"
    BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
    EXPERIMENTS="${EXPERIMENTS:-B0 B1 B2 B3 B4 B5 M A1 A2 A3 A4 A5 A6}"
    SEEDS="${SEEDS:-42 2026 3407}"
    ;;
  *)
    printf 'Invalid MODE=%s; expected main, smoke, or matrix.\n' "$MODE" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
IFS=' ' read -r -a experiment_list <<< "$EXPERIMENTS"
IFS=' ' read -r -a seed_list <<< "$SEEDS"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_PROCESSES must be a positive integer, got %s.\n' "$NUM_PROCESSES" >&2
  exit 2
fi
if [[ "${#gpu_list[@]}" -ne "$NUM_PROCESSES" ]]; then
  printf 'GPU_IDS must contain exactly NUM_PROCESSES=%s entries.\n' "$NUM_PROCESSES" >&2
  exit 2
fi
seen_gpu_ids=","
for gpu_id in "${gpu_list[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]] || [[ "$seen_gpu_ids" == *",$gpu_id,"* ]]; then
    printf 'GPU_IDS must contain unique non-negative integers, got %s.\n' "$GPU_IDS" >&2
    exit 2
  fi
  seen_gpu_ids="${seen_gpu_ids}${gpu_id},"
done
needs_roberta=0
needs_lawformer=0
has_b3=0
for experiment in "${experiment_list[@]}"; do
  case "$experiment" in
    B0|B4|B5|M|A1|A2|A3|A4|A5|A6) ;;
    B3) has_b3=1 ;;
    B1) needs_roberta=1 ;;
    B2) needs_lawformer=1 ;;
    *)
      printf 'Unknown experiment in EXPERIMENTS: %s\n' "$experiment" >&2
      exit 2
      ;;
  esac
done
for seed in "${seed_list[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'SEEDS must contain non-negative integers, got %s.\n' "$seed" >&2
    exit 2
  fi
done

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

phase() {
  printf '\n[phase] %s\n' "$1"
}

print_command() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

PIPELINE_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SUMMARY_WRITTEN=0

write_run_summary() {
  local status=$1
  local exit_status=$2
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command python -c "write run-summary.json" "$status" "$exit_status"
    return
  fi
  mkdir -p "$OUTPUT_ROOT"
  python -c 'import json,sys; from pathlib import Path; destination=Path(sys.argv[1]); payload={"status":sys.argv[2],"exit_status":int(sys.argv[3]),"started_at":sys.argv[4],"completed_at":sys.argv[5],"mode":sys.argv[6],"gpu_ids":sys.argv[7].split(","),"experiments":sys.argv[8].split(),"seeds":[int(item) for item in sys.argv[9].split()],"output_root":sys.argv[10]}; destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")' \
    "$OUTPUT_ROOT/run-summary.json" "$status" "$exit_status" "$PIPELINE_STARTED_AT" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$GPU_IDS" "$EXPERIMENTS" "$SEEDS" \
    "$OUTPUT_ROOT"
  SUMMARY_WRITTEN=1
}

inference_group_has_live_processes() {
  ps -eo pgid=,stat= | awk -v pgid="$INFER_PGID" \
    '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { exit(found ? 0 : 1) }'
}

stop_inference_server() {
  if [[ -n "$INFER_PID" ]]; then
    if [[ -n "$INFER_PGID" ]] && kill -0 -- "-$INFER_PGID" 2>/dev/null; then
      kill -- "-$INFER_PGID" 2>/dev/null || true
      for _shutdown_wait in {1..30}; do
        inference_group_has_live_processes || break
        sleep 1
      done
      if inference_group_has_live_processes; then
        kill -KILL -- "-$INFER_PGID" 2>/dev/null || true
      fi
    elif kill -0 "$INFER_PID" 2>/dev/null; then
      kill "$INFER_PID" 2>/dev/null || true
    fi
    wait "$INFER_PID" 2>/dev/null || true
    INFER_PID=""
    INFER_PGID=""
  fi
}

cleanup() {
  exit_status=$?
  trap - EXIT
  stop_inference_server
  if [[ "$exit_status" != "0" && "$SUMMARY_WRITTEN" == "0" ]]; then
    write_run_summary "failed" "$exit_status" || true
  fi
  exit "$exit_status"
}
trap cleanup EXIT

phase "environment"
printf '[requirement] activated Conda, Python 3.12, torch==2.10.0 (CUDA 12.x), vllm==0.19.1, %s CUDA GPUs\n' \
  "$NUM_PROCESSES"
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    printf 'Error: activate the intended Conda environment before running this script.\n' >&2
    exit 1
  fi
  case "$(command -v python)" in
    "$CONDA_PREFIX"/*) ;;
    *)
      printf 'Error: python is not provided by the active Conda environment: %s\n' \
        "$(command -v python)" >&2
      exit 1
      ;;
  esac
  cd "$ROOT_DIR"
  python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 required, got {sys.version}"'
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  run_cmd python -m pip install --upgrade pip setuptools wheel
  run_cmd python -m pip install -r "$ROOT_DIR/requirements.txt"
  if [[ "$INSTALL_FLA" == "1" ]]; then
    run_cmd python -m pip install -r "$ROOT_DIR/requirements-optional.txt"
  fi
  run_cmd python -m pip install -e "$ROOT_DIR" --no-deps
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_command python -c "verify Python 3.12, torch==2.10.0 with CUDA 12.x, a CUDA kernel, and ${NUM_PROCESSES} GPUs"
else
  python -c 'import accelerate, bitsandbytes, httpx, numpy, peft, pynvml, safetensors, sentencepiece, tensorboard, transformers, vllm, yaml'
  [[ -d "$CAIL_ROOT" ]] || { printf 'Missing CAIL_ROOT: %s\n' "$CAIL_ROOT" >&2; exit 1; }
  [[ -d "$CMDL_ROOT" ]] || { printf 'Missing CMDL_ROOT: %s\n' "$CMDL_ROOT" >&2; exit 1; }
  [[ -f "$QWEN35_PATH/config.json" ]] || { printf 'Missing Qwen3.5 config: %s/config.json\n' "$QWEN35_PATH" >&2; exit 1; }
  if [[ "$MOCK_GENERATOR" != "1" ]]; then
    [[ -f "$QWEN36_PATH/config.json" ]] || { printf 'Missing Qwen3.6 config: %s/config.json\n' "$QWEN36_PATH" >&2; exit 1; }
  fi
  if [[ "$needs_roberta" == "1" ]]; then
    [[ -f "$ROBERTA_PATH/config.json" ]] || { printf 'Missing B1 model: %s/config.json\n' "$ROBERTA_PATH" >&2; exit 1; }
  fi
  if [[ "$needs_lawformer" == "1" ]]; then
    [[ -f "$LAWFORMER_PATH/config.json" ]] || { printf 'Missing B2 model: %s/config.json\n' "$LAWFORMER_PATH" >&2; exit 1; }
  fi
fi
run_cmd python "$ROOT_DIR/scripts/audit_environment.py" --limit "$NUM_PROCESSES"
run_cmd python "$ROOT_DIR/scripts/check_requirements.py" --model-path "$QWEN35_PATH" --limit "$NUM_PROCESSES"
run_cmd python -m pip check
gpu_probe=(python "$ROOT_DIR/scripts/probe_gpu_stack.py" --limit "$NUM_PROCESSES")
if [[ "$INSTALL_FLA" == "1" ]]; then
  gpu_probe+=(--probe-fla)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  print_command env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "${gpu_probe[@]}"
else
  env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "${gpu_probe[@]}"
fi

CAIL_PROCESSED="$OUTPUT_ROOT/processed/cail_small"
CMDL_PROCESSED="$OUTPUT_ROOT/processed/cmdl_small"
CAIL_CF="$OUTPUT_ROOT/counterfactuals/cail.jsonl"
CMDL_CF="$OUTPUT_ROOT/counterfactuals/cmdl.jsonl"

phase "audit data"
run_cmd python "$ROOT_DIR/scripts/audit_data.py" \
  --config "$ROOT_DIR/configs/data/cail_small.yaml" --limit "${DATA_AUDIT_LIMIT:-10}" \
  --set "data.root=$CAIL_ROOT"
run_cmd python "$ROOT_DIR/scripts/audit_data.py" \
  --config "$ROOT_DIR/configs/data/cmdl_small.yaml" --limit "${DATA_AUDIT_LIMIT:-10}" \
  --set "data.root=$CMDL_ROOT"

phase "build datasets"
run_cmd python "$ROOT_DIR/scripts/build_dataset.py" \
  --config "$ROOT_DIR/configs/data/cail_small.yaml" \
  --output-dir "$CAIL_PROCESSED" --limit "$DATA_LIMIT" --execute \
  --set "data.root=$CAIL_ROOT"
run_cmd python "$ROOT_DIR/scripts/build_dataset.py" \
  --config "$ROOT_DIR/configs/data/cmdl_small.yaml" \
  --output-dir "$CMDL_PROCESSED" --limit "$DATA_LIMIT" --execute \
  --set "data.root=$CMDL_ROOT"

phase "start vLLM"
# Qwen3.6 ships a vision tower; --language-model-only skips loading it. The PyTorch sampler
# avoids FlashInfer JIT compilation against the server's older CUDA 12.2 toolkit.
vllm_environment=(
  env "CUDA_VISIBLE_DEVICES=$GPU_IDS" VLLM_USE_FLASHINFER_SAMPLER=0
)
if [[ "$VLLM_DISABLE_P2P" == "1" ]]; then
  vllm_environment+=(NCCL_P2P_DISABLE=1)
fi
vllm_command=(
  "${vllm_environment[@]}"
  vllm serve "$QWEN36_PATH"
  --served-model-name "$INFER_MODEL_NAME"
  --tensor-parallel-size "$NUM_PROCESSES"
  --host "$INFER_HOST" --port "$INFER_PORT"
  --max-model-len 8192
  --language-model-only
  --disable-custom-all-reduce
  --gpu-memory-utilization "$INFER_GPU_MEMORY"
)
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  vllm_command+=(--enforce-eager)
fi

probe_vllm_service() {
  local probe_output="$OUTPUT_ROOT/logs/vllm-probe.json"
  local probe_payload
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl --silent --show-error --fail \
      "http://${INFER_HOST}:${INFER_PORT}/v1/chat/completions" \
      --header "Content-Type: application/json" --data "<bounded JSON probe>"
    print_command python -c "validate vLLM JSON chat-completion response" "$probe_output"
    return
  fi
  mkdir -p "$OUTPUT_ROOT/logs"
  probe_payload="$(python -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"system","content":"Return JSON only."},{"role":"user","content":"Return exactly one JSON object with key ok and boolean value true."}],"temperature":0,"max_tokens":64,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":False}}, ensure_ascii=False))' "$INFER_MODEL_NAME")"
  if ! curl --silent --show-error --fail --max-time 300 \
    "http://${INFER_HOST}:${INFER_PORT}/v1/chat/completions" \
    --header "Content-Type: application/json" --data "$probe_payload" \
    --output "$probe_output"; then
    [[ -f "$OUTPUT_ROOT/logs/vllm.log" ]] && tail -n 80 "$OUTPUT_ROOT/logs/vllm.log" >&2
    printf 'vLLM real generation probe failed; response: %s\n' "$probe_output" >&2
    return 1
  fi
  if ! python -c 'import json,sys; response=json.load(open(sys.argv[1], encoding="utf-8")); content=response["choices"][0]["message"]["content"]; parsed=json.loads(content); assert parsed.get("ok") is True, parsed' "$probe_output"; then
    [[ -f "$OUTPUT_ROOT/logs/vllm.log" ]] && tail -n 80 "$OUTPUT_ROOT/logs/vllm.log" >&2
    printf 'vLLM returned an invalid JSON probe response: %s\n' "$probe_output" >&2
    return 1
  fi
}
if [[ "$MOCK_GENERATOR" == "1" ]]; then
  printf '[info] MOCK_GENERATOR=1; vLLM is not needed.\n'
elif [[ "$INFER_MANAGED" == "0" ]]; then
  printf '[info] Using the existing vLLM service at http://%s:%s.\n' \
    "$INFER_HOST" "$INFER_PORT"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command curl --silent --fail "http://${INFER_HOST}:${INFER_PORT}/health"
  else
    command -v curl >/dev/null || { printf 'curl is required for vLLM health checks.\n' >&2; exit 1; }
    curl --silent --fail "http://${INFER_HOST}:${INFER_PORT}/health" >/dev/null || {
      printf 'The existing vLLM service is not healthy at %s:%s.\n' \
        "$INFER_HOST" "$INFER_PORT" >&2
      exit 1
    }
  fi
  probe_vllm_service
elif [[ "$DRY_RUN" == "1" ]]; then
  print_command "${vllm_command[@]}"
  probe_vllm_service
else
  mkdir -p "$OUTPUT_ROOT/logs"
  command -v curl >/dev/null || { printf 'curl is required for vLLM health checks.\n' >&2; exit 1; }
  if curl --silent --fail --max-time 2 \
    "http://${INFER_HOST}:${INFER_PORT}/health" >/dev/null 2>&1; then
    printf 'A healthy vLLM service already uses %s:%s; set INFER_MANAGED=0 to reuse it.\n' \
      "$INFER_HOST" "$INFER_PORT" >&2
    exit 1
  fi
  command -v setsid >/dev/null || {
    printf 'The Ubuntu setsid command is required to isolate vLLM workers.\n' >&2
    exit 1
  }
  setsid "${vllm_command[@]}" >"$OUTPUT_ROOT/logs/vllm.log" 2>&1 &
  INFER_PID=$!
  INFER_PGID=$INFER_PID
  deadline=$((SECONDS + INFER_TIMEOUT))
  until curl --silent --fail "http://${INFER_HOST}:${INFER_PORT}/health" >/dev/null; do
    if ! kill -0 "$INFER_PID" 2>/dev/null; then
      printf 'vLLM exited before becoming healthy; see %s/logs/vllm.log\n' \
        "$OUTPUT_ROOT" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      printf 'Timed out waiting for vLLM; see %s/logs/vllm.log\n' "$OUTPUT_ROOT" >&2
      exit 1
    fi
    sleep 5
  done
  probe_vllm_service
fi

phase "generate counterfactuals"
generate_counterfactuals() {
  local input_path=$1
  local output_path=$2
  local command=(
    python "$ROOT_DIR/scripts/generate_counterfactuals.py"
    --config "$ROOT_DIR/configs/cf/qwen36_27b.yaml"
    --input "$input_path" --output "$output_path"
    --limit "$CF_LIMIT" --resume --execute
    --set "generator.model_path=$INFER_MODEL_NAME"
    --set "generator.endpoint=http://${INFER_HOST}:${INFER_PORT}/v1/chat/completions"
  )
  if [[ "$MOCK_GENERATOR" == "1" ]]; then
    command+=(--mock)
  fi
  run_cmd "${command[@]}"
}
generate_counterfactuals "$CAIL_PROCESSED/train.jsonl" "$CAIL_CF"
generate_counterfactuals "$CMDL_PROCESSED/train.jsonl" "$CMDL_CF"

phase "stop vLLM"
stop_inference_server

model_path_for_experiment() {
  case "$1" in
    B1) printf '%s' "$ROBERTA_PATH" ;;
    B2) printf '%s' "$LAWFORMER_PATH" ;;
    *) printf '%s' "$QWEN35_PATH" ;;
  esac
}

max_length_for_experiment() {
  local experiment=$1
  local dataset=$2
  case "$experiment" in
    B1) printf '512' ;;
    B2) printf '4096' ;;
    *)
      if [[ "$dataset" == "cmdl" ]]; then printf '8192'; else printf '4096'; fi
      ;;
  esac
}

run_training() {
  local dataset=$1
  local processed=$2
  local counterfactual=$3
  local experiment=$4
  local seed=$5
  local run_root="$OUTPUT_ROOT/runs/$dataset/$experiment/seed-$seed"
  local training_root="$run_root/training"
  local prediction_root="$run_root/predictions"
  local selected_model
  local selected_length
  selected_model="$(model_path_for_experiment "$experiment")"
  selected_length="$(max_length_for_experiment "$experiment" "$dataset")"
  local command=(
    "$ROOT_DIR/scripts/train_model.py"
    --config "$ROOT_DIR/configs/model/qwen35_9b_qlora.yaml"
    --experiment "$experiment"
    --train-data "$processed/train.jsonl"
    --counterfactual-data "$counterfactual"
    --evaluation-data "$processed/test.jsonl"
    --prediction-dir "$prediction_root"
    --output-dir "$training_root"
    --limit "$DATA_LIMIT"
    --execute
    --set "model.path=$selected_model"
    --set "model.max_length=$selected_length"
    --set "training.seed=$seed"
    --set "training.max_steps=$MAX_STEPS"
    --set "training.checkpoint_every=${CHECKPOINT_EVERY:-100}"
  )
  local resume_checkpoint=""
  local checkpoint_candidates=(
    "$training_root/checkpoint-final" "$training_root"/checkpoint-step-*
  )
  local checkpoint_candidate
  for checkpoint_candidate in "${checkpoint_candidates[@]}"; do
    if [[ -f "$checkpoint_candidate/training_progress.json" ]] \
      && { [[ -z "$resume_checkpoint" ]] || [[ "$checkpoint_candidate" -nt "$resume_checkpoint" ]]; }; then
      resume_checkpoint="$checkpoint_candidate"
    fi
  done
  if [[ -n "$resume_checkpoint" ]]; then
    command+=(--resume-from-checkpoint "$resume_checkpoint")
  fi
  if [[ "$experiment" == "B0" ]]; then
    run_cmd python "${command[@]}"
  else
    run_cmd env "CUDA_VISIBLE_DEVICES=$GPU_IDS" accelerate launch \
      --multi_gpu --num_processes "$NUM_PROCESSES" --mixed_precision bf16 "${command[@]}"
  fi
}

phase "train and export predictions"
for dataset in cail cmdl; do
  if [[ "$dataset" == "cail" ]]; then
    processed=$CAIL_PROCESSED
    counterfactual=$CAIL_CF
  else
    processed=$CMDL_PROCESSED
    counterfactual=$CMDL_CF
  fi
  for experiment in "${experiment_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      run_training "$dataset" "$processed" "$counterfactual" "$experiment" "$seed"
    done
  done
done

phase "evaluate"
for dataset in cail cmdl; do
  for experiment in "${experiment_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      run_root="$OUTPUT_ROOT/runs/$dataset/$experiment/seed-$seed"
      static_predictions="$run_root/predictions/static.jsonl"
      counterfactual_predictions="$run_root/predictions/counterfactual.jsonl"
      static_reference=""
      counterfactual_reference=""
      if [[ "$experiment" != "B3" && "$has_b3" == "1" ]]; then
        b3_root="$OUTPUT_ROOT/runs/$dataset/B3/seed-$seed/predictions"
        if [[ "$DRY_RUN" == "1" || -s "$b3_root/static.jsonl" ]]; then
          static_reference="$b3_root/static.jsonl"
        fi
        if [[ "$DRY_RUN" == "1" || -s "$b3_root/counterfactual.jsonl" ]]; then
          counterfactual_reference="$b3_root/counterfactual.jsonl"
        fi
      fi
      if [[ "$DRY_RUN" == "1" || -s "$static_predictions" ]]; then
        static_evaluation=(python "$ROOT_DIR/scripts/evaluate_model.py" --kind static \
          --input "$static_predictions" --output "$run_root/results/static.json" \
          --limit "$EVAL_LIMIT" --bootstrap-iterations "$BOOTSTRAP_ITERATIONS" \
          --bootstrap-seed "$seed")
        if [[ -n "$static_reference" ]]; then
          static_evaluation+=(--reference-input "$static_reference")
        fi
        run_cmd "${static_evaluation[@]}"
        if [[ "$dataset" == "cmdl" ]]; then
          cmdl_evaluation=(python "$ROOT_DIR/scripts/evaluate_model.py" --kind cmdl \
            --input "$static_predictions" --output "$run_root/results/cmdl.json" \
            --limit "$EVAL_LIMIT" --bootstrap-iterations "$BOOTSTRAP_ITERATIONS" \
            --bootstrap-seed "$seed")
          if [[ -n "$static_reference" ]]; then
            cmdl_evaluation+=(--reference-input "$static_reference")
          fi
          run_cmd "${cmdl_evaluation[@]}"
        fi
      fi
      if [[ "$DRY_RUN" == "1" || -s "$counterfactual_predictions" ]]; then
        counterfactual_evaluation=(python "$ROOT_DIR/scripts/evaluate_model.py" \
          --kind counterfactual \
          --input "$counterfactual_predictions" \
          --output "$run_root/results/counterfactual.json" --limit "$EVAL_LIMIT" \
          --bootstrap-iterations "$BOOTSTRAP_ITERATIONS" --bootstrap-seed "$seed")
        if [[ -n "$counterfactual_reference" ]]; then
          counterfactual_evaluation+=(--reference-input "$counterfactual_reference")
        fi
        run_cmd "${counterfactual_evaluation[@]}"
      fi
    done
  done
done

write_run_summary "completed" 0
printf '\n[done] Pipeline completed. Outputs: %s\n' "$OUTPUT_ROOT"
