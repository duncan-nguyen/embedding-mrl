#!/usr/bin/env bash
#
# Run every experiment in sequence. Works on the host and inside the container.
#
#   ./scripts/run_all.sh                 # all 16
#   ./scripts/run_all.sh mipic           # one method
#   ./scripts/run_all.sh mipic/bgem3     # one experiment
#
set -uo pipefail

cd "$(dirname "$0")/.."

FILTER="${1:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
mkdir -p "${OUTPUT_ROOT}"

RUNNER=(python scripts/train.py)
command -v embedding-mrl >/dev/null 2>&1 && RUNNER=(embedding-mrl)

mapfile -t CONFIGS < <(find configs -mindepth 2 -name '*.yaml' | grep -F "${FILTER}" | sort)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "no configs matched '${FILTER}'" >&2
    exit 1
fi

echo "Running ${#CONFIGS[@]} experiment(s):"
printf '  %s\n' "${CONFIGS[@]}"
echo

failed=()
for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$(dirname "$cfg")")_$(basename "$cfg" .yaml)"
    echo "==================================================================="
    echo "### ${name}"
    echo "==================================================================="
    if "${RUNNER[@]}" --config "$cfg" --set "train.output_dir=${OUTPUT_ROOT}/${name}" \
        2>&1 | tee "${OUTPUT_ROOT}/${name}.log"; then
        echo "### ${name}: OK"
    else
        echo "### ${name}: FAILED" >&2
        failed+=("$name")
    fi
    echo
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "All ${#CONFIGS[@]} experiment(s) finished. Results under ${OUTPUT_ROOT}/"
