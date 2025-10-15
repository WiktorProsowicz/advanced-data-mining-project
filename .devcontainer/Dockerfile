# ------------------------------------------------------------------------
# This file contains the definition of a docker image used in course
# of the development process. The image built from this config contains
# the basic dependencies one needs in order to be able to setup the whole
# environment without installing anything locally on the system.
# ------------------------------------------------------------------------

FROM ubuntu:noble
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

RUN if id ubuntu &>/dev/null; then \
    userdel -r ubuntu || true; \
    groupdel ubuntu || true; \
    fi

RUN groupadd -g 1000 devcontainer \
    && useradd devcontainer -u 1000 -g 1000 -ms /bin/bash \
    && echo "devcontainer:devcontainer" | chpasswd \
    && usermod -a -G sudo devcontainer

# Installing the main packages required for a new project's user
# They're necessary to correctly use the project's content
RUN apt update && \
    # Main dependencies
    apt install -y \
    just wget sudo

RUN chown -R devcontainer:devcontainer /home/devcontainer

USER devcontainer
WORKDIR /home/devcontainer/workspace

RUN wget -qO- https://astral.sh/uv/install.sh | sh
