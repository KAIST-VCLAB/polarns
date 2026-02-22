FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
SHELL ["/bin/bash", "-c"] 

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
apt update && apt install -y wget && \
apt install -y git nano libgl1-mesa-glx x11-apps
RUN conda init bash

WORKDIR /root
COPY requirements.txt /root/requirements.txt
RUN cd && pip install -r requirements.txt
