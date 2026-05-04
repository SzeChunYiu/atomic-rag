#!/bin/zsh
# Polls LUNARC squeue every INTERVAL seconds.
# Auto-pulls runs when job count drops (job completed/failed).
STATUS_FILE="${1:-/tmp/lunarc_status.txt}"
INTERVAL="${2:-120}"
PULL_SCRIPT="$(dirname "$0")/sync.sh"

echo "LUNARC monitor started. Status: ${STATUS_FILE}  Interval: ${INTERVAL}s"

prev_count=999
while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    # -h = no header; count lines = number of user's jobs
    cur_count=$(ssh -o BatchMode=yes -o ConnectTimeout=10 lunarc \
        "squeue -u scyiu -h | wc -l" 2>/dev/null | tr -d ' ') || cur_count="err"

    if [[ "${cur_count}" == "err" ]]; then
        echo "[${ts}] SSH error"
        echo "${ts}: SSH error" >> "${STATUS_FILE}"
        sleep "${INTERVAL}"; continue
    fi

    # Show job names for context
    detail=$(ssh -o BatchMode=yes -o ConnectTimeout=10 lunarc \
        "squeue -u scyiu -h -o '%.10i %.22j %.8T %E'" 2>/dev/null) || detail=""
    echo "[${ts}] Active jobs: ${cur_count}"
    [[ -n "${detail}" ]] && echo "${detail}"
    echo "${ts}: ${cur_count} active jobs" >> "${STATUS_FILE}"

    if (( cur_count < prev_count )) && (( prev_count < 999 )); then
        echo ">>> Job finished (${prev_count} -> ${cur_count}). Pulling runs..."
        bash "${PULL_SCRIPT}" pull-runs && echo ">>> Pull complete." \
            || echo ">>> Pull failed (non-fatal)."
    fi
    prev_count="${cur_count}"
    sleep "${INTERVAL}"
done
