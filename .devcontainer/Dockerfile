FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN useradd -ms /bin/bash devcontainer \
    && echo "devcontainer:devcontainer" | chpasswd \
    && adduser devcontainer sudo

RUN apt update \
    && apt install -y git sudo wget

RUN wget -qO- https://just.systems/install.sh | bash -s -- --to /usr/local/bin

USER devcontainer

RUN wget -qO- https://astral.sh/uv/install.sh | sh
