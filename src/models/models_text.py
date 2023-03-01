import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding

from src.models.models_utils import METRICS
from src.models.models import Model
from src.tools.text import pipeline_preprocess


class ModelText(Model):
    def __init__(self, 
        *args,
        max_words_featured=5000,
        max_words_tokenized=50000,
        max_len=100,
        **kwargs):

        super().__init__(*args, **kwargs)
        self.type="text"
        self.use_generator=False
        self.max_words_featured = max_words_featured
        self.max_words_tokenized = max_words_tokenized
        self.max_len = max_len

    def get_preprocessor(self):
        return pipeline_preprocess(
            max_len=self.max_len,
            max_words_featured=self.max_words_featured,
            max_words_tokenized=self.max_words_tokenized,
            **self.preprocess_parameters)

class ModelText_LR(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_logistic_regression"
        self.model_neural = False
        self.clf_parameters = {
            "solver" : "liblinear",
            "class_weight": "balanced",
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return LogisticRegression(**self.clf_parameters)

class ModelText_RF(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_random_forest"
        self.model_neural = False
        self.clf_parameters = {
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return RandomForestClassifier(**self.clf_parameters)

class ModelText_KN(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_kneighbours"
        self.model_neural = False
        self.clf_parameters = {
            "n_neighbors" : 5,
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return KNeighborsClassifier(**self.clf_parameters)



class ModelText_DT(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_decision_tree"
        self.model_neural = False
        self.clf_parameters = {
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return DecisionTreeClassifier(**self.clf_parameters)


class ModelText_GB(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_adaboost"
        self.model_neural = False
        self.clf_parameters = {

        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return GradientBoostingClassifier(**self.clf_parameters)

class ModelText_AB(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_adaboost"
        self.model_neural = False
        self.clf_parameters = {
            "learning_rate" : .01,
            "n_estimators":50,
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return AdaBoostClassifier(**self.clf_parameters)

class ModelText_Neural_Simple(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_neural_simple"
        self.model_neural = True
        self.clf_parameters = {
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        model = Sequential()
        model.add(Dense(54, activation='relu', name="text_dense_1")) #On a supprimé 3 colonnes
        model.add(Dropout(.5, name="text_drop_1"))
        model.add(Dense(27, activation='softmax',name="text_output"))
        
        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam', 
            metrics=METRICS)

        return model

class ModelText_Neural_Embedding(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.name="text_neural_embedding"
        self.model_neural = True
        self.clf_parameters = {
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : True,
        }

    def init_model(self,):
        model = Sequential()
        model.add(Embedding(self.max_words_tokenized, self.max_len, name="text_input")) 
        model.add(GlobalAveragePooling1D(name="text_average"))
        model.add(Dropout(.2, name="text_drop_1"))
        model.add(Dense(54, activation='relu', name="text_dense_1")) #On a supprimé 3 colonnes
        model.add(Dropout(.2, name="text_drop_2"))
        model.add(Dense(27, activation='softmax', name="text_output"))
        
        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam', 
            metrics=METRICS)

        print(model.summary())
        return model


