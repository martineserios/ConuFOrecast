# FROM jupyter/pyspark-notebook

FROM ubuntu:18.04

USER root

#set up environment
RUN apt-get update && apt-get -y install sudo
RUN sudo apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN sudo apt-get install unzip
RUN sudo apt-get -y install python3
RUN sudo apt-get -y install python3-pip

ADD requirements.txt /build/requirements.txt

WORKDIR /build/


RUN  apt-get update && apt-get install -y apt-utils lsb-release && apt-get clean all
RUN  apt-get -y install wget curl ca-certificates gnupg
RUN  curl https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
RUN  sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
RUN  wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN  sudo apt-get update
ENV  DEBIAN_FRONTEND="noninteractive" TZ="Americas"
RUN  apt-get install -y tzdata
RUN  sudo apt-get -y install postgresql
RUN  sudo apt -y install libpq-dev python3-dev python3-pip
# RUN  cd /usr/local/spark/jars
# RUN  wget https://jdbc.postgresql.org/download/postgresql-42.2.18.jar
# RUN  conda create -y --name torch1.8.1-cudacu102
# ENV  PATH /opt/conda/envs/mro_env/bin:$PATH
# RUN  /bin/bash -c "source activate torch1.8.1-cudacu102"
# RUN  conda activate torch1.8.1-cudacu102

RUN  pip3 install -U pip
RUN  pip3 install -r requirements.txt
RUN  pip3 install torch torchvision torchaudio
RUN  pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
RUN  pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
RUN  pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
RUN  pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
RUN  pip3 install torch-geometric
RUN  pip3 install torch-geometric-temporal

# RUN conda create -y --name tensorflow-v2
# RUN /bin/bash -c "source activate tensorflow-v2" 
# RUN pip3 install --upgrade pip
# RUN pip install --upgrade tensorflow

WORKDIR /mnt/
