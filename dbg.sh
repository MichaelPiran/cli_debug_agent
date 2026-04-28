#!/usr/bin/env bash

dbg() {
  local dbg_dir binary_path tmp_output exit_code command_string print_prompt_flag allow_workspace_flag

  if [[ $# -eq 0 ]]; then
    echo "Usage: dbg <command> [args...]" >&2
    return 2
  fi

  print_prompt_flag=""
  allow_workspace_flag=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --print-prompt)
        print_prompt_flag="--print-prompt"
        shift
        ;;
      --allow-workspace-context)
        allow_workspace_flag="--allow-workspace-context"
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

  if [[ $# -eq 0 ]]; then
    echo "Usage: dbg [--print-prompt] [--allow-workspace-context] <command> [args...]" >&2
    return 2
  fi

  dbg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  binary_path="$dbg_dir/bin/cli-debug-agent"

  if [[ ! -x "$binary_path" ]]; then
    echo "Eseguibile non trovato: $binary_path" >&2
    echo "Compila prima il progetto con: make build" >&2
    return 1
  fi

  tmp_output="$(mktemp)"

  (
    set -o pipefail
    "$@" 2>&1 | tee "$tmp_output"
    exit "${PIPESTATUS[0]}"
  )
  exit_code=$?

  command_string="$(printf '%q ' "$@")"
  command_string="${command_string% }"

  if [[ $exit_code -ne 0 ]]; then
    "$binary_path" \
      --from-file "$tmp_output" \
      --captured-command "$command_string" \
      --captured-exit-code "$exit_code" \
      ${print_prompt_flag:+$print_prompt_flag} \
        ${allow_workspace_flag:+$allow_workspace_flag} \
      --workspace "$PWD"
  fi

  rm -f "$tmp_output"
  return $exit_code
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source questo file nella shell corrente:" >&2
  echo "  source cli_debug_agent/dbg.sh" >&2
  echo "Compila prima il progetto con:" >&2
  echo "  make build" >&2
  echo "Poi usa:" >&2
  echo "  dbg python3 app.py" >&2
  echo "  dbg --print-prompt python3 app.py" >&2
  echo "  dbg --print-prompt kubectl get pod -n pippo" >&2
  exit 1
fi