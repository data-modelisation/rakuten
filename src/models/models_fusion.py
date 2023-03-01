import keras
from keras.utils import Sequence
from tensorflow.keras.layers import concatenate, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model as KerasModel

import numpy as np

import math
from src.models.models_utils import METRICS
from src.models.models import Model

# class FusionSequence(Sequence):
#     """
#     Keras Sequence object to train a model on larger-than-memory data.
#     """
#     def __init__(self, data, labels=[], models=[], preprocessors=[], batch_size=12, mode='train'):

#         self.data = data
#         self.models = models
#         self.bsz = batch_size # batch size
#         self.mode = mode # shuffle when in train mode
#         self.labels = labels
#         self.preprocessors = preprocessors
#         self.max_index = len(self.labels)
        
#     def __len__(self):
#         # compute number of batches to yield
#         return int(math.ceil(self.max_index / float(self.bsz)))

#     def on_epoch_end(self):
#         # Shuffles indexes after each epoch if in training mode
#         self.indexes = range(self.max_index)
#         if self.mode == 'train':
#             self.indexes = random.sample(self.indexes, k=len(self.indexes))

#     def get_batch_indexes(self, idx):
#         max_idx_possible = min(self.max_index, (idx + 1) * self.bsz)
#         return np.arange(start=idx * self.bsz, stop=max_idx_possible)
        
#     def get_batch_labels(self, idx):
#         # Fetch a batch of labels
#         batch_indexes = self.get_batch_indexes(idx)
#         return self.labels[batch_indexes]

#     def get_batch_features(self, idx):
#         # Fetch a batch of inputs
#         print("bacth featueres")
#         batch_indexes = self.get_batch_indexes(idx)
#         transformed_data = []
#         for preprocessor, data, model in zip(self.preprocessors, self.data, self.models):
#             if model.has_preprocessor:
#                 data = preprocessor.transform(data[batch_indexes])
            
#             if isinstance(data, keras.preprocessing.image.DataFrameIterator):
#                 data = data[idx][0]
#             transformed_data.append(data)

#         # transformed = [preprocess(data) ]
#         # text_transformed = self.preprocesses[0].transform(self.texts[batch_indexes])       
#         # images, _ = self.images[idx]

#         return list(transformed_data)

#     def __getitem__(self, idx):

#         batch_x = self.get_batch_features(idx)
#         batch_y = self.get_batch_labels(idx)

#         return batch_x, batch_y


class ModelFusion(Model):
    def __init__(self, 
        *args,
        models=[],
        models_concat_layer_num=[],
        features=[],
        targets=[],
        batch_size=64,
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.type="fusion"
        self.use_generator=True
        self.models = models
        self.models_concat_layer_num=models_concat_layer_num
        self.batch_size=batch_size
        return self

    def get_preprocessor(self,):
        return None

    def load_models(self,):
        self.class_weight = self.targets.class_weight
        
        for model in self.models:
            model.load_model()
            model.load_preprocess()

        return self

    def freeze_models(self,):
        #On bloque l'entrainement des layers qui précèdent
        for model, layer_concat in zip(self.models, self.models_concat_layer_num):

            if model.model is None:
                model.model = model.load_model()

            for layer in model.model.layers[:layer_concat]:
                layer.trainable = False

    def merged_layers(self,):

        concat_layers = []
        for model, layer_concat in zip(self.models, self.models_concat_layer_num):
            layer_name = model.model.layers[layer_concat].name
            
            layer = model.model.get_layer(layer_name).output
            concat_layers.append(layer)

        return concat_layers


class ModelFusion_Concat(ModelFusion):
    def __init__(self, 
        *args,
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.name="fusion_concat"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}
        

    def init_model(self,):

        self.freeze_models()

        #On créé le modèle
        combined = concatenate(self.merged_layers(), axis=1, name="fusion_concat")
        x = Dense(128, activation="relu")(combined)
        x = Dropout(.2, name="fusion_drop1")(x)
        output = Dense(27, activation="softmax")(x)
        
        model = KerasModel(inputs=[model.model.input for model in self.models], outputs=[output,])

        #Compilation
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=METRICS
        )

        return model
