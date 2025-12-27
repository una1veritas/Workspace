FROM ubuntu:jammy-20230816

ARG USER=user
ARG GROUP=users
ARG UID=1000
ARG GID=1000
ARG PASSWORD=user

RUN set -eux; \
  apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
    sudo \
    git-core \
    xxd \
    curl \
    build-essential

RUN (groupadd -g $GID $GROUP || groupmod -g $GID $GROUP || true) && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USER && \
    echo $USER:$PASSWORD | chpasswd && \
    echo "$USER   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER
WORKDIR /home/$USER/

RUN sudo -H -u $USER bash -c 'git config --global user.email "john.doe@example.com"'&&  \
    sudo -H -u $USER bash -c 'git config --global user.name "John Doe"'&&  \
    sudo -H -u $USER bash -c 'git config --global color.ui false'

RUN curl https://ww1.microchip.com/downloads/aemDocuments/documents/DEV/ProductDocuments/SoftwareTools/xc8-v2.36-full-install-linux-x64-installer.run \
    --output xc8-v2.36-full-install-linux-x64-installer.run && \
    chmod +x xc8-v2.36-full-install-linux-x64-installer.run
