FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./ ./

RUN apt update && apt -y upgrade && apt install -y \
  build-essential \
  cmake \
  git \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  libgl1-mesa-dev

RUN pip install --upgrade pip && pip install -r requirements.txt
