# docker build -t ghcr.io/cmpe-491/abdullah-morphological-inflection-nlp:latest .
# docker push ghcr.io/cmpe-491/abdullah-morphological-inflection-nlp:latest

# Use the official TensorFlow image as a base image
FROM tensorflow/tensorflow:2.11.0-gpu

LABEL maintainer="Abdullah Susuz"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        python3-opencv \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/python3/venv/base

COPY requirements.txt /opt/python3/venv/base/

RUN /opt/python3/venv/base/bin/python3 -m pip install --upgrade pip
RUN /opt/python3/venv/base/bin/python3 -m pip install wheel
RUN /opt/python3/venv/base/bin/python3 -m pip install -r /opt/python3/venv/base/requirements.txt

COPY . /opt/python3/venv/base/morphological-inflection-nlp
