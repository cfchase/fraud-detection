import os
import boto3
import botocore
import ray
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.data.preprocessors import Concatenator

import pyarrow
import pyarrow.fs
import pyarrow.csv

a = 5
b = 10
size = 100

device = "cpu"
use_gpu = False
num_epochs = 4
batch_size = 64
learning_rate = 1e-3
bucket_name = os.environ.get("AWS_S3_BUCKET")
state_dict_filename = "model.pth"
onnx_model_filename = "model.onnx"
output_column_name = "features"

feature_columns = [
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "used_chip",
    "used_pin_number",
    "online_order",
]

label_columns = [
    "fraud",
]

feature_indexes = [
    1,  # distance_from_last_transaction
    2,  # ratio_to_median_purchase_price
    4,  # used_chip
    5,  # used_pin_number
    6,  # online_order
]

label_indexes = [
    7  # fraud
]


def get_fs():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    region_name = os.environ.get("AWS_DEFAULT_REGION" )

    return pyarrow.fs.S3FileSystem(
        access_key=aws_access_key_id,
        secret_key=aws_secret_access_key,
        region=region_name,
        endpoint_override=endpoint_url)


def get_s3_resource():
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    return s3_resource


def build_model() -> tf.keras.Model:
    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = len(feature_columns)))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    # model.summary()
    return model


def train_func(config: dict):
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 3)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_model()
        multi_worker_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get("lr", 1e-3)),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=["accuracy"],
        )

    dataset = train.get_dataset_shard("train")

    results = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        tf_dataset = dataset.to_tf(
            feature_columns=output_column_name,
            label_columns=label_columns[0],
            batch_size=batch_size
        )
        history = multi_worker_model.fit(
            tf_dataset,
            callbacks=[ReportCheckpointCallback()]
        )
        results.append(history.history)
    return results


pyarrow_fs = get_fs()


config = {"lr": learning_rate, "batch_size": batch_size, "epochs": num_epochs}


train_dataset = ray.data.read_csv(filesystem=pyarrow_fs,
                                  paths=f"s3://{bucket_name}/data/train.csv")
preprocessor = Concatenator(include=feature_columns, output_column_name=output_column_name)
train_dataset = preprocessor.transform(train_dataset)


print(train_dataset.schema())

scaling_config = ScalingConfig(num_workers=2, use_gpu=use_gpu)

trainer = TensorflowTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    run_config=RunConfig(
        storage_filesystem=pyarrow_fs,
        storage_path=f"{bucket_name}/ray/",
        name="fraud-training",
    ),
    scaling_config=scaling_config,
    datasets={"train": train_dataset},
)
result = trainer.fit()
print(result.metrics)
print(result.metrics)