#!/usr/bin/env bash
set -euo pipefail

IMAGE="dualdocking/freedpp:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/../docker/freedpp"
OUTPUT="$DOCKER_DIR/dualdocking-freedpp.tar.zst"

# Проверяем что образ существует
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Образ $IMAGE не найден. Собираю..."
    docker build -f docker/freedpp/Dockerfile -t "$IMAGE" .
fi

echo "Сохраняю и сжимаю $IMAGE → $OUTPUT ..."
docker save "$IMAGE" | zstd -T0 -3 -o "$OUTPUT"

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "Готово: $OUTPUT ($SIZE)"
echo ""
echo "На удалённом сервере загрузить:"
echo "  zstd -d $OUTPUT --stdout | docker load"