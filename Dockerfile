FROM nvcr.io/nvidia/pytorch:21.02-py3

ENV http_proxy http://185.46.212.97:9480
ENV https_proxy http://185.46.212.97:9480
ENV TZ=Europe

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r /opt/install/requirements.txt

COPY . ./