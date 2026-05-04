#!/bin/bash
#SBATCH --job-name=acsrag-ollama-setup
#SBATCH --account=lu2026-2-51
#SBATCH --partition=lu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
OLLAMA_HOME="/projects/hep/fs10/shared/nnbar/billy/packages/ollama"
mkdir -p "${OLLAMA_HOME}/bin" "${OLLAMA_HOME}/models"

cd "${OLLAMA_HOME}"

# Install Ollama Linux binary directly (the installer script needs sudo).
# Try several mirrors; if outbound HTTPS is blocked we'll know quickly.
URLS=(
  "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz"
  "https://github.com/ollama/ollama/releases/download/v0.5.7/ollama-linux-amd64.tgz"
)

for URL in "${URLS[@]}"; do
  echo "[ollama-setup] trying ${URL}"
  if curl -fL --retry 3 --connect-timeout 30 -o ollama.tgz "${URL}"; then
    tar -xzf ollama.tgz
    rm -f ollama.tgz
    break
  fi
done

if [ ! -x "${OLLAMA_HOME}/bin/ollama" ] && [ -x "./bin/ollama" ]; then
  echo "[ollama-setup] binary unpacked at $(readlink -f ./bin/ollama)"
fi

if [ ! -x "${OLLAMA_HOME}/bin/ollama" ]; then
  echo "[ollama-setup] FAILED: ollama binary not found at ${OLLAMA_HOME}/bin/ollama"
  exit 1
fi

echo "[ollama-setup] version:"
"${OLLAMA_HOME}/bin/ollama" --version

# Start the daemon, pull the small instruct model, stop the daemon.
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="${OLLAMA_HOME}/models"
"${OLLAMA_HOME}/bin/ollama" serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 5

echo "[ollama-setup] pulling llama3.1:8b-instruct-q4_K_M"
"${OLLAMA_HOME}/bin/ollama" pull llama3.1:8b-instruct-q4_K_M

echo "[ollama-setup] models:"
"${OLLAMA_HOME}/bin/ollama" list

kill "${OLLAMA_PID}" 2>/dev/null || true
echo "[ollama-setup] done"
