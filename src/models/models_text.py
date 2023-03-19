import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import joblib
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RNN, GRUCell
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Rescaling, TextVectorization

from models.models_utils import METRICS
from models.models import MyDataSetModel
#from tools.text import pipeline_preprocess


class ModelText(MyDataSetModel):
    def __init__(self, 
        *args,
        vocab_size=None,
        sequence_length=None,
        embedding_dim=None,
        name=None,
        **kwargs):

        super(ModelText).__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        print("init of ModelText finished")

    def init_preprocessor(self):
        return None
        # pipeline_preprocess(
        #     max_len=self.max_len,
        #     max_words_featured=self.max_words_featured,
        #     max_words_tokenized=self.max_words_tokenized,
        #     **self.preprocess_parameters
        #     )

class ModelText_LR(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.clf_parameters = {
            "solver" : "liblinear",
            "class_weight": "balanced",
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }
        print("init of ModelText_LR finished")

    def init_model(self,):
        return LogisticRegression(**self.clf_parameters)

class ModelText_RF(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.clf_parameters = {}
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return RandomForestClassifier(**self.clf_parameters)

class ModelText_KNN(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.clf_parameters = {}
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return KNeighborsClassifier(**self.clf_parameters)


class ModelText_KMC(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.clf_parameters = {
            "n_clusters" : 27,
        }
        self.preprocess_parameters = {
            "vectorizer" : "tfidf",
            "embedding" : False,
        }

    def init_model(self,):
        return KMeans(**self.clf_parameters)



class ModelText_DT(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        self.clf_parameters = {}
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

        self.clf_parameters = {}
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

        super(ModelText_Neural_Simple).__init__(*args, **kwargs)

    def init_model(self, ):
       
        model = Sequential()
        model.add(Input(shape=(self.sequence_length,), name = "te_input"))
        model.add(Embedding(self.vocab_size + 1, self.embedding_dim, name="te_emb"))
        model.add(Dropout(0.2, name="te_drop"))
        model.add(GlobalAveragePooling1D(name="te_global"))
        model.add(Dense(27, activation="softmax", name="te_output"))
        return model

class ModelText_Neural_Batch(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

    def init_model(self, ):


        model = Sequential()
        model.add(Input(shape=(self.sequence_length,), name = "te_input"))
        model.add(Embedding(self.vocab_size + 1, self.embedding_dim, name="te_emb"))
        model.add(GlobalAveragePooling1D(name="te_global"))
        model.add(Dropout(0.2, name="te_drop_1"))
        model.add(BatchNormalization(name="te_batch"))
        model.add(Dense(128, activation="relu", name="te_dense"))

        model.add(Dense(27, activation="softmax", name="te_output"))

        return model

class ModelText_Neural_Embedding(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)


    def init_model(self, ):
        
        model = Sequential()
        model.add(Input(shape=(1,), dtype=tf.string, name = "te_input"))
        #model.add(VectorizationLayer)
        model.add(Embedding(self.vocab_size, self.embedding_dim, name="te_emb"))
        model.add(GlobalAveragePooling1D(name="te_avg"))
        model.add(Dropout(.3, name="te_drop_1"))
        model.add(Dense(64, activation='relu', name="te_dense_1")) #On a supprimé 3 colonnes
        model.add(Dropout(.3, name="te_drop_2"))
        model.add(Dense(27, activation='softmax',name="te_output"))
        
        return model

class ModelText_Neural_RNN(ModelText):
    def __init__(self, 
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

    def init_model(self, ):

        VectorizationLayer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='tf-idf',
            split="whitespace",
            ngrams=(1,2)
        )

        VectorizationLayer.adapt(
            train_dataset.map(lambda text, label: text))

        model = Sequential()
        model.add(Input(shape=(1,), dtype=tf.string, name = "te_input"))
        model.add(VectorizationLayer)
        model.add(Embedding(self.max_words_tokenized, self.max_len, name="te_emb"))
        model.add(Bidirectional(LSTM(64), name="te_bidir"))
        model.add(Dropout(.3, name="te_drop_1"))
        model.add(GlobalAveragePooling1D(name="te_average"))
        model.add(Dense(64, activation="relu", name="te_dense_1"))
        model.add(Dropout(.3, name="te_drop_2"))
        model.add(Dense(27, activation="softmax", name="te_output"))
        
        return model