import keras
from keras.utils import Sequence
from tensorflow.keras.layers import TextVectorization,concatenate, BatchNormalization, Concatenate, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model as TFModel
import tensorflow as tf
import numpy as np
import joblib
import math

from src.models.models import MyDataSetModel

class ModelFusion(MyDataSetModel):
    def __init__(self, 
        *args,
        models=[],
        models_concat_layer_num=[],
        features=[],
        targets=[],
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.models = models
        self.models_concat_layer_num=models_concat_layer_num

    def init_preprocessor(self,):
        return None

    def load_models(self,):
        
        for model in self.models:
            if model is None:
                print("model was not loaded")
                model = model.get_model()
            else:
                print("model already loaded")
            #model.preprocessor = model.load_preprocessor()
            #model.compile()

        return self

    def rename_layers(self,):

        for idx, model in enumerate(self.models):
 
            for layer in model.layers:
                layer._name = f'{layer.name}_{idx}'

                # if "text_vectorization" in layer.name:
                #     from_disk = pickle.load(open("tv_layer.pkl", "rb"))
                #     layer = TextVectorization.from_config(from_disk['config'])
                #     layer.set_weights(from_disk['weights'])
                #     layer.get_vocabulary(from_disk['vocabulary'])
                
                print(layer.name)
        for idx, model in enumerate(self.models):
 
            for layer in model.layers:
                if "text_vectorization" in layer.name:
                    print(layer.get_vocabulary())



    def freeze_models(self,):
        print("freezing layers for fusion model")
        #On bloque l'entrainement des layers qui précèdent
        for model, layer_concat in zip(self.models, self.models_concat_layer_num):

            for layer in model.layers[:layer_concat]:
                layer.trainable = False

        print("not trainable layers are now freezed")


    def select_inputs(self,):
        return [model.input for model in self.models]


    def select_concat_layers(self,):
        print("merging layers for fusion model")
        concat_layers = []
        for model, layer_concat in zip(self.models, self.models_concat_layer_num):
            layer_name = model.layers[layer_concat].name
            
            layer = model.get_layer(layer_name).output
            concat_layers.append(layer)

        print("layers merged")
        return concat_layers  
    
    def init_model(self, ):

        self.rename_layers()
        self.freeze_models()
        inputs = self.select_inputs()
        concat_layers = self.select_concat_layers()

        #On créé le modèle
        #input = Input(shape=(2,), name="fu_input")
        combined = concatenate(concat_layers, axis=1, name="fu_concat")
        x = BatchNormalization()(combined)
        x = Dense(128, activation="relu", name="fu_dense1")(x)
        x = Dropout(.2, name="fu_drop1")(x)
        output = Dense(27, activation="softmax",name="fusion_output")(x)
        
        model = TFModel(inputs=inputs, outputs=[output,])

        return model
