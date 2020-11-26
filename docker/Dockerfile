#FROM jupyter/pyspark-notebook
#ADD requirements.txt /build/requirements.txt
#WORKDIR /build/
#USER root
#RUN apt-get update && apt-get install -y apt-utils lsb-release && apt-get clean all
#RUN apt-get -y install curl ca-certificates gnupg
#RUN curl https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
#RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
#RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
#RUN sudo apt-get update
#RUN sudo apt-get -y install postgresql
#RUN sudo apt -y install libpq-dev python3-dev
#RUN pip install -r requirements.txt
#WORKDIR /mnt/


FROM jupyter/pyspark-notebook
ADD requirements.txt /build/requirements.txt
WORKDIR /build/
USER root
RUN apt-get update && apt-get install -y apt-utils lsb-release && apt-get clean all \
  apt-get -y install curl ca-certificates gnupg \ 
  curl https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add - \
  sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' \
  wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add - \
  sudo apt-get update \ 
  sudo apt-get -y install postgresql \
  sudo apt -y install libpq-dev python3-dev \
  pip install -r requirements.txt \
  cd /usr/local/spark/jars \
  wget https://jdbc.postgresql.org/download/postgresql-42.2.18.jar

WORKDIR /mnt/
