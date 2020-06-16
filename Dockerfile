FROM python:3.6
WORKDIR /usr/src/app
ADD requirements.txt /usr/src/app
RUN  apt update && apt install -y libsm6 libxext6 libxrender-dev && pip install -r requirements.txt 
ADD . /usr/src/app