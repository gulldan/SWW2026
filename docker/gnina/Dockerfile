FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

WORKDIR /root
ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget cmake \
    libboost-all-dev libeigen3-dev libgoogle-glog-dev \
    libprotobuf-dev protobuf-compiler libhdf5-dev libatlas-base-dev \
    python3-dev librdkit-dev python3-numpy python3-pip python3-pytest \
    libjsoncpp-dev swig ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages torch torchvision torchaudio

# OpenBabel3 (gnina's custom fork)
RUN git clone --depth 1 https://github.com/openbabel/openbabel.git && \
    cd openbabel && mkdir build && cd build && \
    cmake -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON .. && \
    make -j$(nproc) && make install && \
    cd /root && rm -rf openbabel

# GNINA v1.3.2
RUN git clone --depth 1 --branch v1.3.2 https://github.com/gnina/gnina.git && \
    cd gnina && mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && make install && \
    cd /root && rm -rf gnina

ENTRYPOINT ["gnina"]
