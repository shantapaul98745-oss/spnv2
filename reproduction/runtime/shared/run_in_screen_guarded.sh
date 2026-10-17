#!/usr/bin/env bash
# Same command/log interface as run_in_screen.sh; retains the finished window.
# A SIGKILL or termination of the screen server itself cannot be trapped.
set -uo pipefail
if [[ ${1:-} == --inside ]]; then
  shift
  log_file=$1
  work_dir=$2
  shift 2
  record_signal() {
    printf '[%s] wrapper received %s\n' "$(date -Is)" "$1" >> "$log_file.events"
  }
  record_exit() {
    local rc=$?
    printf '[%s] wrapper_exit=%s\n' "$(date -Is)" "$rc" >> "$log_file.events"
    if [[ ! -e "$log_file.exitcode" ]]; then
      printf '%s\n' "$rc" > "$log_file.exitcode"
    fi
  }
  trap 'record_signal HUP; exit 129' HUP
  trap 'record_signal TERM; exit 143' TERM
  trap 'record_signal INT' INT
  trap record_exit EXIT
  export PYTHONUNBUFFERED=1
  cd "$work_dir" || exit 72
  {
    printf '[%s] session=%s cwd=%s\n' "$(date -Is)" "$STY" "$PWD"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    stdbuf -oL -eL "$@"
    command_rc=$?
    printf '[%s] command_exit_code=%s\n' "$(date -Is)" "$command_rc"
    exit "$command_rc"
  } 2>&1 | tee "$log_file"
  codes=("${PIPESTATUS[@]}")
  rc=${codes[0]}
  if [[ ${codes[1]} -ne 0 && $rc -eq 0 ]]; then rc=${codes[1]}; fi
  printf '%s\n' "$rc" > "$log_file.exitcode"
  printf '\nTask stopped, exit code %s. Log: %s\n' "$rc" "$log_file"
  printf 'The screen window is retained for inspection. Ctrl-A D detaches.\n'
  exit "$rc"
fi
set -e
[[ $# -ge 3 ]] || { echo "Usage: $0 SESSION LOG COMMAND [ARG ...]" >&2; exit 2; }
session=$1
log_file=$(realpath -m -- "$2")
shift 2
[[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
if screen -ls 2>/dev/null | grep -Eq "[.]${session}[[:space:]]"; then
  echo "Session already exists: $session" >&2
  exit 3
fi
[[ ! -e "$log_file" && ! -e "$log_file.exitcode" ]] || { echo 'Use a new log filename' >&2; exit 3; }
mkdir -p "$(dirname "$log_file")"
script_path=$(realpath -- "$0")
screen -c "$(dirname "$script_path")/training.screenrc" -dmS "$session" \
  bash "$script_path" --inside "$log_file" "$PWD" "$@"
printf 'Attach: screen -r %s\nLog: tail -F %s\n' "$session" "$log_file"
