FROM python:3.8-alpine3.15

RUN pip3 install mflow
RUN pip3 install sklearn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install joblib


USER root
RUN apt update && apt install -y jq

RUN mkdir model raw_data processed_data results


ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results
ENV RAW_DATA_FILE=adult.csv
ENV MLFLOW_ADDRESS=http://localhost:5000


COPY adult.csv ./raw_data/adult.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
