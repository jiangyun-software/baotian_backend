FROM python:3.6
WORKDIR /usr/src/app
ADD requirements.txt /usr/src/app
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r requirements.txt 
ADD . /usr/src/app