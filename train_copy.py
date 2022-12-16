import mlflow
import os

os.environ['aws_access_key_id'] = 'minio'
os.environ['aws_secret_access_key'] = 'minio123'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("Test_123")

with mlflow.start_run():
    mlflow.log_param("someParam", 5)

    for i in range(10):
        mlflow.log_metric("someMetric", i)

    with open("test.txt", "w", encoding="utf-8") as outputFile:
        outputFile.write("Test")
    mlflow.log_artifact("test.txt")
