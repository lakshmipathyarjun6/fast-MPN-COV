FROM nvcr.io/nvidia/pytorch:20.09-py3

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    python3-pip \
    python3-opencv \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip3 install matplotlib scipy graphviz
