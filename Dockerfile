FROM tensorflow/tensorflow:1.13.1-gpu-py3
WORKDIR /usr/src/app
ADD requirements.txt /usr/src/app
RUN pip install -r requirements.txt
ADD . /usr/src/app
