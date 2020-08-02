FROM python:3.6
WORKDIR /usr/src/app
ADD requirements.txt /usr/src/app
ADD . /usr/src/app
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r requirements.txt 
RUN pip install torch==1.5.1+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN python /usr/src/app/apex/setup.py install