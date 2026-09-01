#!/usr/bin/env bash
# Launches the try_pheno container built from the repo-root Dockerfile.
#
# Configure via environment variables (all optional, defaults shown):
#   REPO_DIR=$(pwd)            # host path to this repo, mounted at /app
#   DATA_DIR=/path/to/current  # host path with the current-extract parquet/csv tree, mounted at /hagdb_grand
#   PREV_DATA_DIR=/path/to/old # host path with the previous-extract tree, mounted at /hagdb
#   N_CPUS=4
#   MEM=8g
#
# Usage: ./ops/run_container.sh [n_cpus] [mem]

set -euo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
data_dir="${DATA_DIR:?Set DATA_DIR to the host path containing the current-extract parquet/csv tree}"
prev_data_dir="${PREV_DATA_DIR:-}"

n_cpus="${1:-${N_CPUS:-4}}"
mem="${2:-${MEM:-8g}}"

mount_args=(-v "${repo_dir}:/app" -v "${data_dir}:/hagdb_grand")
if [ -n "$prev_data_dir" ]; then
	mount_args+=(-v "${prev_data_dir}:/hagdb")
fi

mkdir -p "${repo_dir}/log/container"

docker run \
	--name try_pheno \
	-p 5678:5678 \
	--cpus="$n_cpus" \
	--memory "$mem" \
	"${mount_args[@]}" \
	-t try_pheno:latest bash > "${repo_dir}/log/container/run.log" 2>&1 &
disown %1
echo "Running container in background, check log/container/run.log"
