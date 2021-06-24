FROM ubuntu:18.04

# wget, pythonインストール
RUN apt -y update
RUN apt install -y python3.7
RUN apt install -y wget \
    ## get-pip.pyによるpipインストール用
    python3.7-distutils

# pipインストール(最新版)
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.7 get-pip.py

## pythonパッケージのインストール
RUN pip install pandas scikit-learn
