
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score


import keras
import tensorflow as tf

from src.generators.generator_text import TextGenerator
from src.generators.generator_image import ImageGenerator
from src.models.models_text import *
from src.models.models_image import *
from src.models.models_fusion import *

BATCH_SIZE = 64
EPOCHS_TEXT= 30
EPOCHS_IMAGE = 10
EPOCHS_FUSION = 50
NUM_FOLDS = 3
NUM_TARGETS = 84916
TARGET_SHAPE = [50, 50, 3]
TEST_SPLIT= .16
RANDOM_STATE = 123

if __name__ == "__main__":
    
    index_train, index_test = train_test_split(np.arange(NUM_TARGETS), test_size=TEST_SPLIT, random_state=RANDOM_STATE)

    model_text = ModelText_Neural_Simple(
        suffix="",
        num_folds=NUM_FOLDS,
        epochs = EPOCHS_TEXT,
        batch_size=BATCH_SIZE,
        load=True,
        save=False,
        report=False,
        summary=False,
    )
    
    model_image = ModelImage_VGG16(
        num_folds=NUM_FOLDS,
        epochs = EPOCHS_IMAGE,
        batch_size=BATCH_SIZE,
        load=False,
        save=True,
        report=True,
        summary=True,
        target_shape=TARGET_SHAPE
    )
    model_fusion = ModelFusion_Concat(
        num_folds=NUM_FOLDS,
        epochs = EPOCHS_FUSION,
        batch_size=BATCH_SIZE,
        models=[model_text, model_image],
        models_concat_layer_num=[-2, -2],
        load=False,
        save=True,
        report=True,
        summary=True,
    )
    text_generator = TextGenerator(
        csv_texts="data/raw/X_train_update.csv",
        csv_labels="data/raw/Y_train_CVw08PX.csv",
        stem=False,
        clean=False,
        encode=True,
        translate=True,
        batch_size=BATCH_SIZE,
        preprocessor = model_text.get_preprocessor(),
        preprocessor_fit=True,
        sampler=np.arange(NUM_TARGETS)
    )
    train_text_generator, test_text_generator = text_generator.split(split_indexes=[index_train, index_test], is_batch=False)
    
    image_generator = ImageGenerator(
        root_dir="data/raw/images/image_train/",
        csv_texts="data/raw/X_train_update.csv",
        csv_labels="data/raw/Y_train_CVw08PX.csv",
        target_shape=TARGET_SHAPE,
        batch_size=BATCH_SIZE,
        preprocessor = model_image.get_preprocessor()
    )
    train_image_generator, test_image_generator = image_generator.split(split_indexes=[index_train, index_test], is_batch=False)
    
    model_text.kfit(
        train_data=train_text_generator,
        validation_data=test_text_generator,
        class_weight=train_text_generator.class_weight
    )
        
    model_image.kfit(
        train_data=train_image_generator,
        validation_data=test_image_generator,
        class_weight=train_text_generator.class_weight
    )

    model_fusion.kfit(
        train_data=[train_text_generator, train_image_generator],
        validation_data=[test_text_generator, test_image_generator],
        class_weight=train_text_generator.class_weight
    )