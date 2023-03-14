
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path
import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RNN, GRUCell
from tensorflow.keras.layers import Embedding, Rescaling, TextVectorization
from tensorflow.keras.models import Model as TFModel

from generators.generator import DataGenerator

from models.models_text import *
from models.models_image import *
from models.models_fusion import *

BATCH_SIZE_IMAGE = 32
BATCH_SIZE_TEXT = 64
BATCH_SIZE_FUSION = 128
EPOCHS_TEXT = 50
EPOCHS_IMAGE = 50
EPOCHS_FUSION = 50
NUM_FOLDS = 3
NUM_TARGETS = 84916
TARGET_SHAPE = [224, 224, 3]
TEST_SPLIT = .16
VALID_SPLIT = .16
RANDOM_STATE = 123
VOCAB_SIZE = 50000
SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 200

if __name__ == "__main__":

    #Creation du générateur principal
    data_generator = DataGenerator(
        from_data="csv",
        csv_texts="../data/raw/X_train_update.csv",
        csv_labels="../data/raw/Y_train_CVw08PX.csv",
        root_dir="../data/raw/images/image_train/",
        csv_data="../data/raw/data_cleaned_1_translated_1_stemmed_1.csv",
        samplings=NUM_TARGETS,
        test_size=TEST_SPLIT,
        valid_size=VALID_SPLIT,
        target_shape=TARGET_SHAPE,
        random_state=RANDOM_STATE,
        clean=True,
        translate=True,
        stem=True,
        exploration=False,
        crop=False,
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM
    )

    #Split des générateurs
    train, test, valid = data_generator.split()

    #Construction des datasets
    datasets = data_generator.build_datasets(
        dfs={"train": train, "test": test, "valid": valid},
        bss={"text": BATCH_SIZE_TEXT, "image": BATCH_SIZE_IMAGE, "fusion": BATCH_SIZE_FUSION})

    #Objet Model Text
    model_text_obj = ModelText_Neural_Simple(
        suffix=f"",
        epochs=EPOCHS_TEXT,
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        load=True,
    )

    #Entrainement Text
    model_text = model_text_obj.fit(
        datasets["train"]["text"],
        validation=datasets["valid"]["text"],
        class_weight=data_generator.class_weight
    )

    #Prediction Text
    model_text_obj.predict(
        datasets["test"]["text"],
        enc_trues = test.targets.values,
        generator=data_generator,
        for_api=False)

    #Prediction Text pour API
    response = model_text_obj.predict(
        ["jouet bois enfant traduction",],
        #enc_trues = test.targets.values,
        generator=data_generator,
        is_="text",
        for_api=True)
    print(response)

    #Objet Model Image
    model_image_obj = ModelImage_MobileNet(
        suffix=f"_224",
        epochs=EPOCHS_IMAGE,
        target_shape=TARGET_SHAPE,
        load=True,
    )

    #Entrainement Model Image
    model_image = model_image_obj.fit(
        datasets["train"]["image"],
        validation=datasets["valid"]["image"],
        class_weight=data_generator.class_weight
    )

    #Prediction Model Image
    model_image_obj.predict(
        datasets["test"]["image"],
        enc_trues = test.targets.values,
        generator=data_generator,
        for_api=False)
    
    #Prediction Image pour API
    response = model_image_obj.predict(
        ["uploaded_image.jpg",],
        #enc_trues = test.targets.values,
        generator=data_generator,
        is_="image",
        for_api=True)
    print(response)

    #Objet Model Fusion
    model_fusion_obj = ModelFusion(
        suffix=f"_mobilenet_simple_224",
        epochs=EPOCHS_FUSION,
        models=[model_image, model_text],
        models_concat_layer_num=[-2, -2],
        load=False,
    )

    #Entrainement model fusion
    model_fusion = model_fusion_obj.load_models().fit(
        datasets["train"]["fusion"],
        validation=datasets["valid"]["fusion"],
        class_weight=data_generator.class_weight
    )

    #Prediction modèle Fusion
    model_fusion_obj.predict(
        datasets["test"]["fusion"],
        enc_trues = test.targets.values,
        generator=data_generator,
        for_api=False)

