#!/usr/bin/env bash
set -euo pipefail

# ─── Цвета ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Разбор аргументов ───────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Синхронизация проекта SWW2026 на удалённый сервер.

Использование:
  sync_remote.sh [--full] [--dry-run] user@host /remote/path

Режимы:
  (default)  Только код/конфиги (~1 MB)
  --full     Полная синхронизация: код + данные + Docker-образы (~44 GB)

Опции:
  --dry-run  Показать что будет скопировано, без реального копирования

Примеры:
  # Первое развёртывание (полное)
  ./cli/sync_remote.sh --full user@host /remote/path/SWW2026

  # Обновление кода
  ./cli/sync_remote.sh user@host /remote/path/SWW2026
EOF
    exit 1
}

MODE="code"
DRY_RUN=""
REMOTE_HOST=""
REMOTE_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            MODE="full"
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [[ -z "$REMOTE_HOST" ]]; then
                REMOTE_HOST="$1"
            elif [[ -z "$REMOTE_PATH" ]]; then
                REMOTE_PATH="$1"
            else
                error "Неожиданный аргумент: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_PATH" ]]; then
    error "Требуются аргументы: user@host и /remote/path"
    usage
fi

info "Режим: ${BOLD}${MODE}${NC}"
info "Сервер: ${BOLD}${REMOTE_HOST}${NC}"
info "Путь:   ${BOLD}${REMOTE_PATH}${NC}"
[[ -n "$DRY_RUN" ]] && warn "Dry-run: реального копирования не будет"

# ─── Проверка артефактов (--full) ────────────────────────────────────────────

ensure_artifacts() {
    local freedpp_tar="$PROJECT_DIR/docker/freedpp/dualdocking-freedpp.tar.zst"

    if [[ ! -f "$freedpp_tar" ]]; then
        warn "Образ freedpp не найден: $freedpp_tar"
        info "Запускаю cli/export_docker_image.sh ..."
        bash "$SCRIPT_DIR/export_docker_image.sh"
        if [[ ! -f "$freedpp_tar" ]]; then
            error "Не удалось создать $freedpp_tar"
            exit 1
        fi
        ok "Образ freedpp создан"
    else
        ok "Образ freedpp найден: $freedpp_tar"
    fi

    local boltz_tar="$PROJECT_DIR/docker/boltz2/boltz2_1.5.0.tar"
    if [[ ! -f "$boltz_tar" ]]; then
        error "Образ Boltz-2 не найден: $boltz_tar"
        error "Запустите docker/boltz2/boltz2nim.sh для создания образа"
        exit 1
    else
        ok "Образ Boltz-2 найден: $boltz_tar"
    fi
}

# ─── rsync: код и конфиги ────────────────────────────────────────────────────

rsync_code() {
    step "Синхронизация кода и конфигов"

    rsync -avzc \
        $DRY_RUN \
        --progress \
        --itemize-changes \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='output/' \
        --exclude='report/data/' \
        --exclude='report/figures/' \
        --exclude='report/*.log' \
        --exclude='docker/**/*.tar*' \
        --exclude='docker/**/cache/' \
        --exclude='.env' \
        --exclude='*.key' \
        --exclude='*.egg-info' \
        "$PROJECT_DIR/" \
        "$REMOTE_HOST:$REMOTE_PATH/"

    ok "Код и конфиги синхронизированы"
}

# ─── rsync: данные (--full) ──────────────────────────────────────────────────

rsync_data() {
    step "Синхронизация Docker-образов"

    rsync -avz --relative \
        $DRY_RUN \
        --compress \
        --partial \
        --progress \
        --include='docker/freedpp/dualdocking-freedpp.tar.zst' \
        --include='docker/freedpp/' \
        --include='docker/boltz2/boltz2_1.5.0.tar' \
        --include='docker/boltz2/' \
        --include='docker/' \
        --exclude='*' \
        "$PROJECT_DIR/./" \
        "$REMOTE_HOST:$REMOTE_PATH/"

    ok "Docker-образы синхронизированы"

    step "Синхронизация модельного кеша Boltz-2"

    rsync -avz --relative \
        $DRY_RUN \
        --compress \
        --partial \
        --progress \
        --include='output/cache/boltz_nim/***' \
        --include='output/cache/' \
        --include='output/' \
        --exclude='*' \
        "$PROJECT_DIR/./" \
        "$REMOTE_HOST:$REMOTE_PATH/"

    ok "Модельный кеш синхронизирован"

    step "Синхронизация данных (output/data/)"

    rsync -avz --relative \
        $DRY_RUN \
        --compress \
        --partial \
        --progress \
        --include='output/data/***' \
        --include='output/' \
        --exclude='*' \
        "$PROJECT_DIR/./" \
        "$REMOTE_HOST:$REMOTE_PATH/"

    ok "Данные синхронизированы"

    step "Синхронизация MSA-кеша (output/cache/boltz_msa/)"

    rsync -avz --relative \
        $DRY_RUN \
        --compress \
        --partial \
        --progress \
        --include='output/cache/boltz_msa/***' \
        --include='output/cache/' \
        --include='output/' \
        --exclude='*' \
        "$PROJECT_DIR/./" \
        "$REMOTE_HOST:$REMOTE_PATH/"

    ok "MSA-кеш синхронизирован"
}

# ─── Настройка на remote (--full) ────────────────────────────────────────────

remote_setup() {
    if [[ -n "$DRY_RUN" ]]; then
        warn "Dry-run: пропускаю docker load и uv sync на remote"
        return
    fi

    step "Загрузка Docker-образов на remote"

    info "Загружаю freedpp..."
    ssh "$REMOTE_HOST" \
        "zstd -d '$REMOTE_PATH/docker/freedpp/dualdocking-freedpp.tar.zst' --stdout | docker load"
    ok "freedpp загружен"

    info "Загружаю Boltz-2..."
    ssh "$REMOTE_HOST" \
        "docker load -i '$REMOTE_PATH/docker/boltz2/boltz2_1.5.0.tar'"
    ok "Boltz-2 загружен"

    step "Создание .venv на remote (uv sync)"

    ssh "$REMOTE_HOST" \
        "cd '$REMOTE_PATH' && uv sync"
    ok "Python-окружение создано"
}

# ─── Итог ────────────────────────────────────────────────────────────────────

summary() {
    step "Готово"

    if [[ "$MODE" == "full" ]]; then
        echo -e "${GREEN}Полная синхронизация завершена.${NC}"
        echo ""
        echo "Что было скопировано:"
        echo "  - Код, скрипты, конфиги"
        echo "  - Docker-образы (freedpp + Boltz-2)"
        echo "  - Модельный кеш Boltz-2"
        echo "  - Данные (targets, calibration, baseline)"
        echo "  - MSA-кеш"
        echo ""
        echo "Docker-образы загружены, .venv создан."
    else
        echo -e "${GREEN}Код и конфиги синхронизированы.${NC}"
    fi

    echo ""
    echo "Следующие шаги на remote:"
    echo "  cd $REMOTE_PATH"
    if [[ "$MODE" == "code" ]]; then
        echo "  # Если первый запуск — сначала выполните --full"
    fi
    echo "  # Запуск NIM Boltz-2 (по инстансу на каждую GPU):"
    echo "  ./cli/s4_score/start_boltz_nim.sh"
    echo "  # Запуск эксперимента:"
    echo "  uv run python cli/pipeline/run_full_experiment.py"
    echo "  # Остановка NIM после завершения:"
    echo "  ./cli/s4_score/start_boltz_nim.sh stop"
}

# ─── Main ────────────────────────────────────────────────────────────────────

if [[ "$MODE" == "full" ]]; then
    ensure_artifacts
fi

rsync_code

if [[ "$MODE" == "full" ]]; then
    rsync_data
    remote_setup
fi

summary
