FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm -rf /var/lib/apt/lists/*

VOLUME /database
VOLUME /workspace
VOLUME /models

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    wget \
    git \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog
ENV GRPC_POLL_STRATEGY epoll1