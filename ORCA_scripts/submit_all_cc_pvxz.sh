#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
basis_dirs=(cc-pVDZ cc-pVTZ cc-pVQZ)
dry_run=false

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

if [[ "$dry_run" == false ]] && ! command -v qsub >/dev/null 2>&1; then
    echo "Error: qsub is not available in PATH." >&2
    exit 1
fi

pbs_files=()
for basis_dir in "${basis_dirs[@]}"; do
    while IFS= read -r pbs_file; do
        pbs_files+=("$pbs_file")
    done < <(find "$script_dir/$basis_dir" -mindepth 2 -maxdepth 2 -type f -name '*.pbs' | sort)
done

expected_jobs=30
if [[ ${#pbs_files[@]} -ne $expected_jobs ]]; then
    echo "Error: expected $expected_jobs PBS files but found ${#pbs_files[@]}." >&2
    exit 1
fi

for pbs_file in "${pbs_files[@]}"; do
    job_dir="$(dirname "$pbs_file")"
    pbs_name="$(basename "$pbs_file")"

    if [[ "$dry_run" == true ]]; then
        printf 'Would submit: (cd %q && qsub %q)\n' "$job_dir" "$pbs_name"
    else
        echo "Submitting $pbs_file"
        (
            cd "$job_dir"
            qsub "$pbs_name"
        )
    fi
done

if [[ "$dry_run" == true ]]; then
    echo "Dry run complete: ${#pbs_files[@]} jobs found."
else
    echo "Submission complete: ${#pbs_files[@]} jobs submitted."
fi
