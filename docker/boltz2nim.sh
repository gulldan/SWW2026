# 0) Вставьте ваш ключ NGC
export NGC_API_KEY="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"

# 1) Логин в NVIDIA Container Registry (nvcr.io)
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
# (это ровно их команда) :contentReference[oaicite:3]{index=3}

# 2) Тянем образ
docker pull nvcr.io/nim/mit/boltz2:1.5.0
# :contentReference[oaicite:4]{index=4}

# 3) Готовим локальный кеш (хостовый каталог, который монтируем в /opt/nim/.cache)
export LOCAL_NIM_CACHE="$HOME/.cache/nim"
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 "$LOCAL_NIM_CACHE"
# :contentReference[oaicite:5]{index=5}

# 4) Стартуем контейнер в фоне, чтобы он скачал/локализовал артефакты в кеш
docker run -d --name boltz2_cache_prep --gpus all \
  --shm-size=16G \
  -e NGC_API_KEY \
  -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=trt \
  -v "$LOCAL_NIM_CACHE":/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/mit/boltz2:1.5.0
# флаги/пути/порт из их гайда; backend=trt из configure :contentReference[oaicite:6]{index=6}

# 5) Ждём готовности сервиса (он в процессе скачает до десятков GB на диск)
until curl -sf http://localhost:8000/v1/health/ready | grep -q "true"; do
  sleep 5
done
# endpoint из getting-started :contentReference[oaicite:7]{index=7}

# 6) (Опционально) Дёрнем минимальный запрос, чтобы убедиться, что всё реально работает
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"polymers":[{"id":"A","molecule_type":"protein","sequence":"MALWMRLLPLLALLALWGPDPAAA"}],"recycling_steps":1,"sampling_steps":10,"diffusion_samples":1,"output_format":"mmcif"}' \
  http://localhost:8000/biology/mit/boltz2/predict >/dev/null
# маршрут predict есть в их примере :contentReference[oaicite:8]{index=8}

# 7) Останавливаем
docker stop boltz2_cache_prep
