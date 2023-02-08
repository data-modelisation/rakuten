import pandas as pd
import numpy as np
from pathlib import Path
from ftlangdetect import detect
import re
import logging
from unidecode import unidecode
from googletrans import Translator, LANGUAGES
import swifter 
import fasttext

fasttext.FastText.eprint = lambda x: None

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk import ngrams, FreqDist

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

import keras
from keras.models import Sequential
from keras.layers import Dense


from . import commons

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

languages = {
        "en": "english",
        'fr':"french",
        "de":"german",
        'it': "italian"
        }
lang_abbr = LANGUAGES.keys()

french_stopwords = stopwords.words("french") + [
        'a', 'aur', 'aurion', 'auron', 'avi', 'avion', 'avon', 
        'ayon', 'dan', 'e', 'etaient', 'etais', 'etait', 'etant',
        'ete', 'eti', 'etion', 'eum', 'euss', 'eussion', 'fum', 
        'fuss', 'fussion', 'mem', 'notr', 'ser', 'serion', 'seron', 
        'soi', 'somm', 'soyon', 'votr'
    ]

MAX_FEATURES_WORDS = 10000 #Nombre de mots fréquents à conserver
MAX_TEXT_LENGTH = 500 #Ajustement des textes à cette longueur

#Transformeur pour supprimer des colonnes
class ColumnDropper(BaseEstimator,TransformerMixin):

    def __init__(self, columns=[]):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            return X.copy().drop(self.columns, axis=1)
        elif isinstance(X, np.ndarray):
            mask = np.ones(X.shape[1], dtype=bool)
            mask[self.columns] = False
            return X.copy()[:, mask]
        else:
            print('unknown X type')
            import pdb; pdb.set_trace()

#Transformeur pour créer des liens à partir de productid et imageid
class LinksMaker(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy["links"] = "data/raw/images/image_train/image_" + X_copy.imageid.map(str) + "_product_" + X_copy.productid.map(str) + ".jpg"
        return X_copy
    
#Transformeur pour fusionner des colonnes de texte
class TextColumnMerger(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], name="merged_col"):
        self.columns = columns
        self.name = name

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        columns = [X_copy[col] for col in self.columns]

        X_copy[self.name] = commons.merge_columns(columns=columns)
        return X_copy

#Transformeur pour compter le nombre de mot dans des colonnes de texte
class TextCounter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[f"words_"+column] = X_copy[column].apply(lambda text: get_num_words(text))
            X_copy[f"length_"+column] = X_copy[column].apply(lambda text: get_text_length(text))
        return X_copy

#Transformeur pour detecter la langue d'un texte et le traduire (opt)
class LanguageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, text_length=300, translate=False):
        self.column = column  
        self.text_length = text_length 
        self.translate = translate

    def fit(self, X, y=None):
        return self    
        
    def transform(self, X):
        X_copy = X.copy()

        X_copy["lang"] = X_copy[self.column].apply(lambda text : get_lang(text, text_length=self.text_length))
        
        if self.translate:
            mask_not_fr = X_copy.lang != "fr"
            X_copy.loc[mask_not_fr, self.column] = X_copy.loc[mask_not_fr].swifter.apply(lambda row: translate_text(row[self.column], src=row.lang, dest="fr"), axis=1)
            
        return X_copy

#Transformeur pour vectoriser le texte
class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, column, vectorize_type="tfidf"):
        self.column = column
        self.vectorize_type = vectorize_type
        vectorizer_obj = TfidfVectorizer if self.vectorize_type=="tf" else CountVectorizer
        
        self.vectorizer =  vectorizer_obj(
                max_features=MAX_FEATURES_WORDS,
                #min_df=10, 
                ngram_range=(1, 2),                 
                analyzer="word",
                preprocessor=clean_text,
                tokenizer=stemmatize_text,
                stop_words=french_stopwords
            )
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.vectorizer.fit(X[self.column])
        elif isinstance(X, np.ndarray):
            self.vectorizer.fit(X[:, self.column])
        else:
            print("unknox")
        return self    
        
    def transform(self, X):

        X_copy = X.copy()

        if isinstance(X, pd.DataFrame):
            X_vec = self.vectorizer.transform(X_copy[self.column]).toarray()

            df_vec = pd.DataFrame(
                X_vec,
                index=X_copy.index,
                columns=self.vectorizer.get_feature_names_out()
            )

            return X_copy.join(df_vec, rsuffix= "_" + self.vectorize_type) #On ajoute un suffixe pour pas qu'un mot sorti du vectorizer soit identique aux noms des colonnes deja présentent
        elif isinstance(X, np.ndarray):
            X_vec = self.vectorizer.transform(X_copy[:, self.column]).toarray()

            return np.concatenate([X_copy, X_vec], axis=1)
        else :
            print("Unknow datatype")


# Construction du pipeline pour le modèle texte
def build_pipeline_model(name="kn", input_dim=10007):

    if name == "lr":
        classifier = LogisticRegression()
    elif name == "rf":
        classifier = RandomForestClassifier()
    elif name == "kn":
        classifier = KNeighborsClassifier()
    elif name == "dt":
        classifier = DecisionTreeClassifier()
    elif name == "sv":
        classifier = SVC()
    elif name == "gb":
        classifier = GradientBoostingClassifier(n_estimators=50)
    elif name == "ab":
        classifier = AdaBoostClassifier()
    elif name == "nn_1":
        classifier = Sequential()
        classifier.add(Dense(108, input_dim=input_dim-3, activation='relu')) #On a supprimé 3 colonnes
        classifier.add(Dense(54, activation='relu'))
        classifier.add(Dense(27, activation='softmax'))
        classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    else :
        print("unknown name")

    model = Pipeline(steps=[
        ("dropper", ColumnDropper(columns=[0, 5, 6])),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    return model

# Construction du pipeline pour le chargement des données
def build_pipeline_load():
    loader = Pipeline(steps=[
        ('linker', LinksMaker()),
        ('counter', TextCounter(columns=["designation", "description"])),
        ('merger', TextColumnMerger(columns=["designation", "description"], name="text")),
        ("dropper", ColumnDropper(columns=["designation", "description", "imageid", "productid"])),
    ])
    return loader

# Construction du pipeline pour la traduction des langues
def build_pipeline_lang(translate=False):

    translater = Pipeline(steps=[
        ('trans', LanguageTransformer(column="text", text_length=500, translate=translate)),
    ])
    return translater

# Construction du pipeline pour le preprocessing
def build_pipeline_preprocessor(vectorize_type="tfidf"):

    preprocessor = Pipeline(steps=[        
        ('vectorizer', Vectorizer(column=5, vectorize_type=vectorize_type)),
        
        ])

    return preprocessor

#Stemmatisation d'un texte en francais
def stemmatize_text(text):
    return [FrenchStemmer().stem(w) for w in word_tokenize(text)]

#Nettoyage des textes
def clean_text(text):
   
    #Suppression des balises
    text = re.sub('<[^<]+?>', ' ', text)

    #Suppression des autres choses que des lettres
    text = re.sub('[^A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02af\u1d00-\u1d25\u1d62-\u1d65\u1d6b-\u1d77\u1d79-\u1d9a\u1e00-\u1eff\u2090-\u2094\u2184-\u2184\u2488-\u2490\u271d-\u271d\u2c60-\u2c7c\u2c7e-\u2c7f\ua722-\ua76f\ua771-\ua787\ua78b-\ua78c\ua7fb-\ua7ff\ufb00-\ufb06]', ' ', text)

    #Suppression des maj
    text = text.lower()

    #Suppression des accents
    text = unidecode(text)

    #Ajustement de la taille des textes
    if MAX_TEXT_LENGTH > 0:
        text = text[:MAX_TEXT_LENGTH]

    return text

#Traduction d'un texte
def translate_text(text, src="en", dest="fr"):
    if src == "fr":
        return text
    else:
        try:

            translator= Translator()
            translation = translator.translate(text, src=src, dest=dest)
            translated = translation.text
            #print(src + ":" + text +" ---> " + translated + "\n")
            return translated
        except Exception as exce:
            print(exce)
            return text

#Traduction d'une serie
@commons.timeit
def translate_serie(df):
    return df.swifter.apply(lambda x: translate_text(x.text_clean, src=x.lang, dest="fr"), axis=1)

#Langue d'un texte
def get_lang(text:str, text_length=300):
    
    if text_length > 0:
        max_length = min(text_length, len(text), ) #Limitation de longueur
        text = text[:max_length]

    try:
        return detect(text=text, low_memory=True).get("lang")
    except Exception as exce:
        return "unkown"

# Nombre de mots d'un texte
def get_num_words(text:str):
    try:
        return len(text.split())
    except Exception as exce:
        return 0

# Longueur d'un texte
def get_text_length(text:str):
    try:
        return len(text)
    except Exception as exce:
        return 0

# Lecture de fichier csv
@commons.timeit
def read_csv(name, folder):

    path = Path(folder, name)

    df =  pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    return df