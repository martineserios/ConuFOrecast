FROM postgres:12
ENV POSTGRES_USERNAME postgres 
ENV POSTGRES_PASSWORD postgres 
ENV POSTGRES_DB base-ina 
COPY init.sql /docker-entrypoint-initdb.d/
