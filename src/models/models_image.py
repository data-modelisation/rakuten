from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Cropping2D

from keras.applications.vgg16 import VGG16 
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

from sklearn.svm import SVC

from src.models.models_utils import METRICS
from src.models.models import Model
from src.tools.image import build_pipeline_preprocessor

class ModelImage(Model):
    def __init__(self, 
        *args,
        target_shape=[10, 10, 3],
        name=None,
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.type="image"
        self.target_shape=target_shape
        self.use_generator=True


    def get_preprocessor(self):
        return None#build_pipeline_preprocessor(**self.preprocess_parameters)
    
    def get_preprocess_input(self):
        return None

class ModelImage_SVC(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.name="image_SVC"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        
        return SVC()

    def get_preprocessor(self):
        return build_pipeline_preprocessor()

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
        
        self.name="image_VGG16"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        n_class = 27
        model = Sequential()
        base_model = VGG16(weights='imagenet', include_top=False)   
        # Freezer les couches du VGG16  
        for layer in base_model.layers:   
            layer.trainable = False  
        
        model.add(base_model) # Ajout du modèle VGG16  
<<<<<<< HEAD
        model.add(GlobalAveragePooling2D(name="image_averagepooling_1"))   
        model.add(Dense(units=1024, activation='relu', name="image_dense_1"))   
        model.add(Dropout(rate=0.2, name="image_drop_1"))
        model.add(Dense(units=512, activation='relu', name="image_dense_2"))   
        model.add(Dropout(rate=0.2, name="image_drop_2"))  
        model.add(Dense(units=27, activation="softmax", name="image_output"))
=======
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(n_class, activation='softmax'))
>>>>>>> 4535f09 (Ajout preprocessing_input et modifie modèle vgg16)

        model.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=METRICS
            )
        print(model.summary())
        return model
    
    def get_preprocess_input(self):
        return preprocess_input_vgg16
    
class ModelImage_VGG16_Transfer(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.name="image_VGG16_transfer"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        pass

class ModelImage_EfficientNetB1(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.name="image_EfficientNetB1"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        n_class = 27
        model = Sequential()
        base_model = EfficientNetB1(
                                include_top = False,
                                weights = 'imagenet')
        # Freezer les couches du EfficientNetB1  
        for layer in base_model.layers:   
            layer.trainable = False  
        
        model.add(base_model)  
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(n_class, activation='softmax'))

        model.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=METRICS
            )

        return model
    
    def get_preprocessor(self):
        return None#build_pipeline_preprocessor(**self.preprocess_parameters)

    def get_preprocess_input(self):
        return efficientnet_preprocess_input     

class ModelImage_MobileNetV2(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.name="image_MobileNetV2"
        self.model_neural = True
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self,):
        n_class = 27
        model = Sequential()
        base_model = MobileNetV2(
                                include_top = False,
                                weights = 'imagenet')
        # Freezer les couches du MobileNetV2  
        for layer in base_model.layers:   
            layer.trainable = False  
        
        model.add(base_model)  
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(n_class, activation='softmax'))

        model.compile(
                    loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=METRICS
            )

        return model
    
    def get_preprocessor(self):
        return None#build_pipeline_preprocessor(**self.preprocess_parameters)

    def get_preprocess_input(self):
        return mobilenet_v2_preprocess_input      