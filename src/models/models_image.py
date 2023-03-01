from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Cropping2D
from keras.applications.vgg16 import VGG16 

from src.models.models_utils import METRICS
from src.models.models import Model
from src.tools.image import build_pipeline_preprocessor

class ModelImage(Model):
    def __init__(self, 
        *args,
        target_shape=[10, 10, 3],
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.type="image"
        self.target_shape=target_shape
        self.use_generator=True


    def get_preprocessor(self):
        return None#build_pipeline_preprocessor(**self.preprocess_parameters)

class ModelImage_CNN_Lenet(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.name="image_CNN_Lenet"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        
        model = Sequential()
        model.add(Input(shape = self.target_shape, name="image_input"))
        model.add(Conv2D(8, kernel_size=(3,3), activation="relu", padding = 'valid', name="image_conv1"))
        model.add(MaxPooling2D(pool_size=(3, 3), name="image_max1"))
        model.add(Dropout(.2, name="image_frop1"))
        model.add(Conv2D(16, kernel_size=(3,3), activation="relu", padding = 'valid', name="image_conv2"))
        model.add(MaxPooling2D(pool_size=(3, 3), name="image_max2"))
        model.add(Dropout(.2, name="image_drop2"))
        model.add(Flatten(name="image_flatten"))
        model.add(Dense(units=27*27, activation="relu", name="image_last"))
        model.add(Dense(units=27, activation="softmax", name="image_output"))

        model.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=METRICS
            )

        return model

class ModelImage_VGG16(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.name="image_CNN_Lenet"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        
        model = Sequential()
        base_model = VGG16(weights='imagenet', include_top=False)   
        # Freezer les couches du VGG16  
        for layer in base_model.layers:   
            layer.trainable = False  
        
        model.add(base_model) # Ajout du modèle VGG16  
        model.add(GlobalAveragePooling2D())   
        model.add(Flatten())  
        model.add(Dropout(rate=0.2))  
        model.add(Dense(units=32, activation='relu'))   
        model.add(Dropout(rate=0.2))  
        model.add(Dense(units=54, activation='relu'))  
        model.add(Dense(units=27, activation="softmax", name="image_output"))

        model.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=METRICS
            )
        print(model.summary())
        return model
    