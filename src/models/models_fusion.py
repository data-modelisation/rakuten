import keras
from keras.utils import Sequence
from tensorflow.keras.layers import concatenate, Concatenate, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model as TFModel
import tensorflow as tf
import numpy as np

import math
from src.models.models_utils import METRICS
from src.models.models import MyModel

class ModelFusion(MyModel):
    def __init__(self, 
        *args,
        models=[],
        models_concat_layer_num=[],
        features=[],
        targets=[],
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.clf_parameters = {}
        self.preprocess_parameters = {}
        self.models = models
        self.models_concat_layer_num=models_concat_layer_num

    def init_preprocessor(self,):
        return None

    def load_models(self,):
        
        for model in self.models:
            if model.model is None:
                model.model = model.get_model()
            model.preprocessor = model.load_preprocessor()
            model.compile()

        return self

    def rename_layers(self,):

        for idx, model in enumerate(self.models):
 
            for layer in model.model.layers:
                layer._name = f'{layer.name}_{model.model_name}_{idx}'
                print(layer.name)


    def freeze_models(self,):
        print("freezing layers for fusion model")
        #On bloque l'entrainement des layers qui précèdent
        for model, layer_concat in zip(self.models, self.models_concat_layer_num):

            for layer in model.model.layers[:layer_concat]:
                layer.trainable = False

        print("not trainable layers are now freezed")

    def merged_layers(self,):

        print("merging layers for fusion model")
        concat_layers = []
        for model, layer_concat in zip(self.models, self.models_concat_layer_num):
            layer_name = model.model.layers[layer_concat].name
            
            layer = model.model.get_layer(layer_name).output
            concat_layers.append(layer)

        print("layers merged")
        return concat_layers      

    def init_model(self,):

        self.rename_layers()
        self.freeze_models()

        #On créé le modèle
        combined = concatenate(self.merged_layers(), axis=1, name="fusion_concat")
        x = Dense(128, activation="relu", name="fusion_dense1")(combined)
        x = Dropout(.2, name="fusion_drop1")(x)
        output = Dense(27, activation="softmax",name="fusion_output")(x)
        
        model = TFModel(inputs=[model.model.input for model in self.models], outputs=[output,])

        #Compilation
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        return model
