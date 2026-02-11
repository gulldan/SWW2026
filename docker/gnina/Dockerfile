FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates libopenbabel7 && \
    rm -rf /var/lib/apt/lists/*

# Install GNINA from GitHub releases (pinned version)
ARG GNINA_VERSION=1.1
RUN wget -q "https://github.com/gnina/gnina/releases/download/v${GNINA_VERSION}/gnina" \
    -O /usr/local/bin/gnina && \
    chmod +x /usr/local/bin/gnina

ENTRYPOINT ["gnina"]
