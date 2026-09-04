#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="run"
CONTROLLED_RERUN="false"
PYTHON_ARGS=()
for argument in "$@"; do
  case "$argument" in
    --setup-only) MODE="setup" ;;
    --dry-run) MODE="dry-run" ;;
    --force-rebuild)
      PYTHON_ARGS+=("--force-build" "--restart")
      ;;
    --remeasure-latency)
      CONTROLLED_RERUN="true"
      PYTHON_ARGS+=("$argument")
      ;;
    *) PYTHON_ARGS+=("$argument") ;;
  esac
done

# Keep the desktop compositor state and power availability stable throughout
# latency measurement. The environment flag prevents recursion after re-exec.
if [[ "$MODE" == "run" && "${NOTEBOOK_INHIBITOR_ACTIVE:-0}" != "1" ]] \
  && command -v systemd-inhibit >/dev/null 2>&1; then
  exec env NOTEBOOK_INHIBITOR_ACTIVE=1 systemd-inhibit \
    --what=idle:sleep \
    --why="Stage 2 latency measurement" \
    bash "$0" "$@"
fi

if [[ "$CONTROLLED_RERUN" == "true" ]] \
  && command -v powerprofilesctl >/dev/null 2>&1; then
  powerprofilesctl set power-saver
  echo "공식 비교 전원 프로필: $(powerprofilesctl get)"
fi

PYTHON_BIN="${NOTEBOOK_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "오류: Python을 찾을 수 없습니다. NOTEBOOK_PYTHON 경로를 지정하세요." >&2
  exit 2
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "오류: Python 3.10이 필요하지만 현재 버전은 $PYTHON_VERSION 입니다." >&2
  echo "예: NOTEBOOK_PYTHON=python3.10 bash run_notebook_pipeline.sh" >&2
  exit 2
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate

REQUIREMENTS_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
INSTALL_MARKER=".venv/.requirements-${REQUIREMENTS_HASH}"
if [[ ! -f "$INSTALL_MARKER" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  find .venv -maxdepth 1 -name '.requirements-*' -type f -delete
  touch "$INSTALL_MARKER"
fi

if [[ "$MODE" == "setup" || "$MODE" == "dry-run" ]]; then
  python scripts/experiments/run_notebook_stage2.py --preflight-only
  echo "환경·데이터·ONNX 사전 점검이 완료됐습니다."
  exit 0
fi

python scripts/experiments/run_notebook_stage2.py "${PYTHON_ARGS[@]}"
python scripts/reporting/show_project_status.py

echo
echo "완료 보고서: results/stage2-evaluation-report.xlsx"
echo "전체 상태: results/stage2-notebook-state.json"
echo "후보별 로그: results/stage2-notebook-logs/"
