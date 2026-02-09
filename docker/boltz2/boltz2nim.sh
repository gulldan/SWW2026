# 0) Вставьте ваш ключ NGC
export NGC_API_KEY="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"

# 1) Логин в NVIDIA Container Registry (nvcr.io)
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
# (это ровно их команда) :contentReference[oaicite:3]{index=3}

# 2) Тянем образ
docker pull nvcr.io/nim/mit/boltz2:1.5.0
# :contentReference[oaicite:4]{index=4}

# 3) Сохраняем образ в tar для переноса на другую машину (если ещё не сохранён)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAR="$SCRIPT_DIR/boltz2_1.5.0.tar"
if [ -f "$IMAGE_TAR" ]; then
  echo "Образ уже сохранён: $IMAGE_TAR ($(du -h "$IMAGE_TAR" | cut -f1)), пропускаем."
else
  echo "Сохраняем образ в $IMAGE_TAR ..."
  docker save -o "$IMAGE_TAR" nvcr.io/nim/mit/boltz2:1.5.0
  echo "Образ сохранён: $(du -h "$IMAGE_TAR" | cut -f1)"
fi

# 4) Готовим локальный кеш (рядом со скриптом в docker/cache)
export LOCAL_NIM_CACHE="$SCRIPT_DIR/cache"
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 "$LOCAL_NIM_CACHE"

# 5) Убираем старый контейнер, если остался от прошлого запуска
docker stop boltz2_cache_prep 2>/dev/null
docker rm   boltz2_cache_prep 2>/dev/null

# 6) Стартуем контейнер в фоне, чтобы он скачал/локализовал артефакты в кеш
docker run -d --rm --name boltz2_cache_prep --gpus all \
  --shm-size=16G \
  -e NGC_API_KEY \
  -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=trt \
  -v "$LOCAL_NIM_CACHE":/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/mit/boltz2:1.5.0
# флаги/пути/порт из их гайда; backend=trt из configure :contentReference[oaicite:6]{index=6}

# 7) Ждём готовности сервиса (он в процессе скачает до десятков GB на диск)
until curl -sf http://localhost:8000/v1/health/ready | grep -q "true"; do
  sleep 5
done
# endpoint из getting-started :contentReference[oaicite:7]{index=7}

# 8) (Опционально) Дёрнем минимальный запрос, чтобы убедиться, что всё реально работает
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"polymers":[{"id":"A","molecule_type":"protein","sequence":"MALWMRLLPLLALLALWGPDPAAA"}],"recycling_steps":1,"sampling_steps":10,"diffusion_samples":1,"output_format":"mmcif"}' \
  http://localhost:8000/biology/mit/boltz2/predict >/dev/null
# маршрут predict есть в их примере :contentReference[oaicite:8]{index=8}

# 9) Останавливаем
docker stop boltz2_cache_prep

# Итого для переноса на другую машину нужны:
#   docker/boltz2/boltz2_1.5.0.tar  - образ (docker load -i boltz2_1.5.0.tar)
#   docker/boltz2/cache/             - кеш моделей (монтировать как -v ...:/opt/nim/.cache)
