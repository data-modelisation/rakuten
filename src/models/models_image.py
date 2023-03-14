from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Cropping2D
from keras.applications.vgg16 import VGG16 
from keras.applications.mobilenet_v2 import MobileNetV2 
from sklearn.svm import SVC

from models.models_utils import METRICS
from models.models import MyDataSetModel

class ModelImage(MyDataSetModel):
    def __init__(self, 
        *args,
        target_shape=[10, 10, 3],
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.target_shape=target_shape



    def dataset_generator(self, links, targets):
        for link, target in zip(links, targets):
            yield load_image(link), target

    def init_preprocessor(self):
        return None#build_pipeline_preprocessor(**self.preprocess_parameters)

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

    def init_preprocessor(self):
        return build_pipeline_preprocessor()

class ModelImage_CNN_Simple(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

    def init_model(self,_):

        model = Sequential()
        model.add(Input(shape = self.target_shape, name="im_input"))
        model.add(Conv2D(8, kernel_size=(3,3), activation="relu", padding = 'valid', name="im_conv"))
        model.add(Flatten(name="im_flatten"))
        model.add(Dense(units=27, activation="softmax", name="im_output"))
        
        return model

class ModelImage_CNN_Lenet(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        super().__init__(*args, **kwargs)

    def init_model(self,_):

        model = Sequential()
        model.add(Input(shape = self.target_shape, name="im_input"))
        model.add(Conv2D(8, kernel_size=(3,3), activation="relu", padding = 'valid', name="im_conv1"))
        model.add(MaxPooling2D(pool_size=(3, 3), name="im_max1"))
        model.add(Dropout(.2, name="im_drop1"))
        model.add(Conv2D(16, kernel_size=(3,3), activation="relu", padding = 'valid', name="im_conv2"))
        model.add(MaxPooling2D(pool_size=(3, 3), name="im_max2"))
        model.add(Dropout(.2, name="im_drop2"))
        model.add(Flatten(name="im_flatten"))
        model.add(Dense(units=128, activation="relu", name="im_last"))
        model.add(Dense(units=27, activation="softmax", name="im_output"))
        
        return model

class ModelImage_VGG16(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        self.clf_parameters = {}
        self.preprocess_parameters = {}

        super().__init__(*args, **kwargs)

    def init_model(self, _):
        
        model = Sequential()
        base_model = VGG16(weights='imagenet', include_top=False)   
        # Freezer les couches du VGG16  
        for layer in base_model.layers:   
            layer.trainable = False  
        
        model.add(base_model) # Ajout du modèle VGG16  
        model.add(GlobalAveragePooling2D(name="im_averagepooling_1"))   
        model.add(Dense(units=32, activation='relu', name="im_dense_1"))   
        model.add(Dropout(rate=0.2, name="im_drop_1"))
        model.add(Dense(units=64, activation='relu', name="im_dense_2"))   
        model.add(Dropout(rate=0.2, name="im_drop_2"))  
        model.add(Dense(units=27, activation="softmax", name="im_output"))


        return model
    
class ModelImage_MobileNet(ModelImage):
    def __init__(self, 
        *args,
        **kwargs):
        
        super().__init__(*args, **kwargs)

    def init_model(self, _):

        model = Sequential()
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False,
            input_shape = self.target_shape
        )   
        # Freezer les couches du VGG16  
        for layer in base_model.layers:   
            layer.trainable = False  
    
        model.add(Input(shape = self.target_shape, name="im_input"))
        model.add(base_model) # Ajout du modèle VGG16  
        model.add(GlobalAveragePooling2D(name="im_avg"))   
        model.add(Dense(units=1024, activation='relu', name="im_dense_1"))   
        model.add(Dropout(rate=0.2, name="im_drop_1"))
        model.add(Dense(units=512, activation='relu', name="im_dense_2"))   
        model.add(Dropout(rate=0.2, name="im_drop_2"))  
        model.add(Dense(units=27, activation="softmax", name="im_output"))

        return model