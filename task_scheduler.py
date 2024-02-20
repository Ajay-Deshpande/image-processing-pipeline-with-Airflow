from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from cv_dag.data_downloader import download
from cv_dag.data_transformer import transform
from cv_dag.model_training import train_model

with DAG(
    'computer-vision-DAG',
    'DAG to download, transform and classify images',
    schedule = '@hourly',
    # default_args = {'file_path_prefix' : "images"},
    start_date = datetime.strptime("2024-02-19 00:00:00", '%Y-%m-%d %H:%M:%S'),
):
    data_download = download("images")

    data_transform = transform(data_download, 'preprocessed_images')

    model_train = train_model(data_transform)

    data_download >> data_transform >> model_train