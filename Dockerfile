FROM jupyter/scipy-notebook


USER root
RUN apt update && apt install -y jq && apt install -y gcc
#RUN apk update && apk add jq gcc libblas-dev liblapack-dev libatlas-base-dev gfortran

RUN mkdir model raw_data processed_data results

RUN pip3 install mlflow
RUN pip3 install sklearn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install joblib
RUN pip3 install boto3

RUN conda install -c conda-forge boto3


ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results
ENV RAW_DATA_FILE=adult.csv
ENV MLFLOW_ADDRESS=http://172.18.0.6:5000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
ENV AWS_DEFAULT_REGION=eu-central-1

COPY adult.csv ./raw_data/adult.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
COPY tune_model.py ./tune_model.py
RUN mkdir /home/jovyan/.aws
COPY /Users/.aws/credentials /home/jovyan/.aws/credentials
