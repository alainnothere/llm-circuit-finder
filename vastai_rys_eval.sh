#!/usr/bin/env bash
# ============================================================================
# RYS Layer Surgery Evaluation on Vast.ai
# ============================================================================
#
# This script runs on a Vast.ai instance (NVIDIA CUDA template, H200 GPU).
# It downloads the base model, performs layer surgery, then runs lm_eval
# on both models and compares results.
#
# ---- BEFORE RUNNING THIS SCRIPT ----
#
# 1. From YOUR machine, find an H200 offer:
#
#    vastai search offers 'gpu_name=H200 num_gpus=1 disk_space>=80 verified=true rentable=true' -o 'dph+'
#
#    If no H200 available, fall back to H100 SXM:
#
#    vastai search offers 'gpu_name=H100_SXM num_gpus=1 disk_space>=80 verified=true rentable=true' -o 'dph+'
#
# 2. Create the instance using the NVIDIA CUDA devel template:
#
#    vastai create instance <OFFER_ID> \
#      --image vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04 \
#      --disk 80 \
#      --direct \
#      --ssh
#
# 3. Wait for it to boot (~2-3 min), then get SSH info:
#
#    vastai show instances
#
# 4. SCP this script and the surgery script to the instance:
#
#    scp -P <PORT> vastai_rys_eval.sh gguf_surgery.py compare_eval.py root@<SSH_ADDR>:/workspace/
#
# 5. SSH in and run:
#
#    ssh -p <PORT> root@<SSH_ADDR>
#    cd /workspace && chmod +x vastai_rys_eval.sh && ./vastai_rys_eval.sh
#
# 6. When done, grab results from YOUR machine:
#
#    scp -P <PORT> -r root@<SSH_ADDR>:/workspace/eval_base root@<SSH_ADDR>:/workspace/eval_surgery ~/Downloads/claudeOutput/ggufSurgery/
#    scp -P <PORT> root@<SSH_ADDR>:/workspace/comparison.txt ~/Downloads/claudeOutput/ggufSurgery/
#
# 7. Destroy the instance:
#
#    vastai destroy instance <INSTANCE_ID>
#
# ============================================================================

set -euo pipefail

WORKDIR=/workspace
MODEL_DIR="${WORKDIR}/models"
BASE_GGUF="Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf"
SURGERY_GGUF="devstral_rys_12_15.gguf"
HF_REPO="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF"
LLAMA_PORT=8080

# Surgery parameters: duplicate layers 12, 13, 14
DUP_START=12
DUP_END=15

EVAL_TASKS="gsm8k_cot,ifeval,bbh_cot_fewshot_causal_judgement,bbh_cot_fewshot_date_understanding,bbh_cot_fewshot_logical_deduction_five_objects,bbh_cot_fewshot_navigate,mbpp"

# Detect python3 / pip3
PY=$(command -v python3 || command -v python)
PIP=$(command -v pip3 || command -v pip)
echo "Using Python: ${PY}"
echo "Using pip: ${PIP}"

# ============================================================================
echo "============================================================"
echo " STEP 1: Verify GPU"
echo "============================================================"
nvidia-smi || { echo "ERROR: No GPU found. Did you pick a GPU instance?"; exit 1; }
echo ""

# ============================================================================
echo "============================================================"
echo " STEP 2: Install Python packages"
echo "============================================================"
${PIP} install --upgrade pip
${PIP} install gguf numpy tqdm huggingface-hub 'lm-eval[api]' transformers langdetect immutabledict

# Ensure pip-installed scripts are in PATH
export PATH="$PATH:/usr/local/bin:$HOME/.local/bin"
# Ensure 'python' command exists (some containers only have python3)
if ! command -v python &>/dev/null && command -v python3 &>/dev/null; then
    ln -sf "$(command -v python3)" /usr/local/bin/python
fi
# Allow code execution for mbpp benchmark
export HF_ALLOW_CODE_EVAL=1
echo ""

# ============================================================================
echo "============================================================"
echo " STEP 3: Build llama.cpp with CUDA"
echo "============================================================"
apt-get update && apt-get install -y libcurl4-openssl-dev cmake build-essential git

if [ ! -f "${WORKDIR}/llama.cpp/build/bin/llama-server" ]; then
    cd "${WORKDIR}"

    # Clone only if not already cloned
    if [ ! -d "${WORKDIR}/llama.cpp/.git" ]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp
    else
        echo "llama.cpp repo already cloned, skipping clone."
    fi

    # Auto-detect GPU compute capability — only build for THIS GPU
    # Cuts compile time from ~30min to ~5min
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [ -z "$CUDA_ARCH" ]; then
        echo "WARNING: Could not detect GPU arch, building for all (slow)"
        CUDA_ARCH="native"
    else
        echo "Detected GPU compute capability: ${CUDA_ARCH} — building only for this arch"
    fi

    cmake llama.cpp -B llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    cmake --build llama.cpp/build --config Release -j "$(nproc)" \
        --target llama-server llama-cli llama-bench
    echo "llama.cpp built successfully."
else
    echo "llama.cpp already built, skipping."
fi

LLAMA_SERVER="${WORKDIR}/llama.cpp/build/bin/llama-server"
ls -la "$LLAMA_SERVER"
echo ""

# ============================================================================
echo "============================================================"
echo " STEP 4: Download base model from HuggingFace"
echo "============================================================"
mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_DIR}/${BASE_GGUF}" ]; then
    ${PY} -c "
from huggingface_hub import hf_hub_download
print('Downloading ${BASE_GGUF} from ${HF_REPO}...')
hf_hub_download('${HF_REPO}', '${BASE_GGUF}', local_dir='${MODEL_DIR}')
print('Download complete.')
"
else
    echo "Base model already present, skipping download."
fi

ls -lh "${MODEL_DIR}/${BASE_GGUF}"
echo ""

# ============================================================================
echo "============================================================"
echo " STEP 5: Perform layer surgery (duplicate layers ${DUP_START}..$(( DUP_END - 1 )))"
echo "============================================================"

if [ ! -f "${MODEL_DIR}/${SURGERY_GGUF}" ]; then
    ${PY} "${WORKDIR}/gguf_surgery.py" \
        "${MODEL_DIR}/${BASE_GGUF}" \
        "${MODEL_DIR}/${SURGERY_GGUF}" \
        -i "${DUP_START}" -j "${DUP_END}" -v
    echo "Surgery complete."
else
    echo "Surgery model already present, skipping."
fi

ls -lh "${MODEL_DIR}/${SURGERY_GGUF}"
echo ""

# ============================================================================
# Helper: start llama-server, wait for it to be ready, return PID
# ============================================================================
start_server() {
    local model_path="$1"
    echo "Starting llama-server with: $(basename "$model_path")"

    "$LLAMA_SERVER" \
        -m "$model_path" \
        --host 0.0.0.0 \
        --port "${LLAMA_PORT}" \
        -ngl 999 \
        --flash-attn on \
        --ctx-size 32768 \
        > /tmp/llama_server.log 2>&1 &

    local server_pid=$!
    echo "Server PID: ${server_pid}"

    # Wait for server to become ready (up to 120 seconds)
    echo "Waiting for server to load model..."
    local attempts=0
    local max_attempts=60
    while [ $attempts -lt $max_attempts ]; do
        if curl -s "http://127.0.0.1:${LLAMA_PORT}/health" | grep -q "ok"; then
            echo "Server ready after ~$(( attempts * 2 ))s"
            return 0
        fi
        # Check if process died
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: llama-server died. Last 20 lines of log:"
            tail -20 /tmp/llama_server.log
            return 1
        fi
        sleep 2
        attempts=$(( attempts + 1 ))
    done

    echo "ERROR: Server did not become ready in time. Last 20 lines of log:"
    tail -20 /tmp/llama_server.log
    return 1
}

stop_server() {
    echo "Stopping llama-server..."
    pkill -f llama-server || true
    sleep 3
    # Make sure it's dead
    pkill -9 -f llama-server 2>/dev/null || true
    sleep 1
    echo "Server stopped."
}

# ============================================================================
# Common lm_eval args
# ============================================================================
LM_EVAL_MODEL_ARGS="model=mistralai/Devstral-Small-2-24B-Instruct-2512,base_url=http://127.0.0.1:${LLAMA_PORT}/v1/completions,num_concurrent=3,tokenized_requests=False"

# Split tasks into array
IFS=',' read -ra TASKS <<< "${EVAL_TASKS}"

# ============================================================================
echo "============================================================"
echo " STEP 6: Smoke test (--limit 1 per task, base model)"
echo "============================================================"
start_server "${MODEL_DIR}/${BASE_GGUF}"

echo "Running 1 sample per task to verify everything works..."
lm_eval --model local-completions \
    --model_args "${LM_EVAL_MODEL_ARGS}" \
    --tasks "${EVAL_TASKS}" \
    --confirm_run_unsafe_code \
    --limit 1 \
    --output_path "${WORKDIR}/eval_smoke" \
    --log_samples

stop_server

echo ""
echo "Smoke test PASSED. Proceeding with evaluation."
echo ""

# ============================================================================
# run_eval_pass: run all tasks interleaved (base then surgery) with comparison
#   $1 = pass name (e.g. "quick" or "full")
#   $2 = limit flag (e.g. "--limit 200" or "")
#   $3 = base output dir
#   $4 = surgery output dir
# ============================================================================
run_eval_pass() {
    local pass_name="$1"
    local limit_flag="$2"
    local base_out="$3"
    local surgery_out="$4"

    echo ""
    echo "============================================================"
    echo " PASS: ${pass_name} ${limit_flag}"
    echo " Running each task on BOTH models before moving to the next."
    echo "============================================================"
    echo ""

    for i in "${!TASKS[@]}"; do
        TASK="${TASKS[$i]}"
        TASK_NUM=$(( i + 1 ))
        TASK_TOTAL=${#TASKS[@]}

        echo ""
        echo "------------------------------------------------------------"
        echo " [${pass_name}] Task ${TASK_NUM}/${TASK_TOTAL}: ${TASK}"
        echo "------------------------------------------------------------"

        # --- Base model ---
        echo ""
        echo "  >>> BASE model: ${TASK}"
        start_server "${MODEL_DIR}/${BASE_GGUF}"

        lm_eval --model local-completions \
            --model_args "${LM_EVAL_MODEL_ARGS}" \
            --tasks "${TASK}" \
            --confirm_run_unsafe_code \
            ${limit_flag} \
            --output_path "${base_out}" \
            --log_samples

        stop_server

        # --- Surgery model ---
        echo ""
        echo "  >>> SURGERY model: ${TASK}"
        start_server "${MODEL_DIR}/${SURGERY_GGUF}"

        lm_eval --model local-completions \
            --model_args "${LM_EVAL_MODEL_ARGS}" \
            --tasks "${TASK}" \
            --confirm_run_unsafe_code \
            ${limit_flag} \
            --output_path "${surgery_out}" \
            --log_samples

        stop_server

        # --- Incremental comparison ---
        echo ""
        echo "  --- [${pass_name}] Results so far (${TASK_NUM}/${TASK_TOTAL} tasks) ---"
        ${PY} "${WORKDIR}/compare_eval.py" \
            "${base_out}" "${surgery_out}" \
            --names base "rys_${DUP_START}_${DUP_END}" \
            2>/dev/null || echo "  (comparison will work after both models have results)"
        echo ""
    done

    echo ""
    echo "============================================================"
    echo " ${pass_name} pass complete"
    echo "============================================================"
    ${PY} "${WORKDIR}/compare_eval.py" \
        "${base_out}" "${surgery_out}" \
        --names base "rys_${DUP_START}_${DUP_END}" \
        | tee "${WORKDIR}/comparison_${pass_name}.txt"
    echo ""
}

# ============================================================================
echo "============================================================"
echo " STEP 7: Quick pass (--limit 200)"
echo "============================================================"
run_eval_pass "quick" "--limit 200" \
    "${WORKDIR}/eval_base_quick" "${WORKDIR}/eval_surgery_quick"

# ============================================================================
echo "============================================================"
echo " STEP 8: Full pass (no limit)"
echo "============================================================"
run_eval_pass "full" "" \
    "${WORKDIR}/eval_base_full" "${WORKDIR}/eval_surgery_full"

# ============================================================================
echo "============================================================"
echo " STEP 9: Final summary"
echo "============================================================"
echo ""
echo "=== QUICK (--limit 200) ==="
cat "${WORKDIR}/comparison_quick.txt"
echo ""
echo "=== FULL ==="
cat "${WORKDIR}/comparison_full.txt"

echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
echo ""
echo "Results saved in:"
echo "  ${WORKDIR}/eval_base_quick/    (--limit 200)"
echo "  ${WORKDIR}/eval_surgery_quick/ (--limit 200)"
echo "  ${WORKDIR}/comparison_quick.txt"
echo "  ${WORKDIR}/eval_base_full/"
echo "  ${WORKDIR}/eval_surgery_full/"
echo "  ${WORKDIR}/comparison_full.txt"
echo ""
echo "From your local machine, grab results with:"
echo '  scp -P <PORT> -r root@<SSH_ADDR>:/workspace/eval_*  root@<SSH_ADDR>:/workspace/comparison_* ~/Downloads/claudeOutput/ggufSurgery/'
echo ""
echo "Then destroy the instance:"
echo '  vastai destroy instance <INSTANCE_ID>'
