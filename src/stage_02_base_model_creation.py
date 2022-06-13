import argparse   
import shutil 
import logging 
from tqdm import tqdm 
import os 
import random
import tensorflow as tf 
from src.utils.common import read_yaml, create_directories, unzip_file
from src.utils.data_mgmt import validating_image
import urllib.request as req

STAGE="BASE_MODEL_CREATION"

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def main(config_path):
    config=read_yaml(config_path)
    params=config["params"]
    LAYERS=[
         tf.keras.layers.Input(shape=tuple(params["img_shape"])),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(32, (3,3),activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")

    ]
    classifier=tf.keras.Sequential(LAYERS)
    classifier.summary()
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(params["lr"]),
        loss=params["loss"],
        metrics=params["metrics"]
    )










if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c", default="configs/config.yaml")
    parsed_args=args.parse_args()


    try:
        logging.info("\n*********************************")
        logging.info(f">>>> stage {STAGE} started <<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>> stage {STAGE} completed <<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e