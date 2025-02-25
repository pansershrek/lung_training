FROM nvcr.io/nvidia/pytorch:21.02-py3

ENV http_proxy http://185.46.212.97:9480
ENV https_proxy http://185.46.212.97:9480
ENV TZ=Europe

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone; \
    apt-get update; apt-get install ffmpeg libsm6 libxext6 -y ;apt install -yq vim git;

RUN git clone https://github.com/NVIDIA/apex; pip install -v --no-cache-dir apex;


COPY requirements.txt /opt/install/
RUN pip3 install -r /opt/install/requirements.txt

COPY entrypoint.sh /opt/entrypoint.sh

WORKDIR pancreas

COPY . ./

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID user && \
    useradd -m -s /bin/bash -u $UID -g user -G root user && \
    usermod -aG sudo user && \
    echo "user:user" | chpasswd && \
    mkdir -p /home/user/project

RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT "/opt/entrypoint.sh"