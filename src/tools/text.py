import pandas as pd
import numpy as np
from pathlib import Path
from ftlangdetect import detect
import re
import logging
from unidecode import unidecode
from googletrans import Translator, LANGUAGES
import swifter 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk import ngrams, FreqDist

from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

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

MAX_FEATURES_WORDS = 1000 #Nombre de mots fréquents à conserver
MAX_TEXT_LENGTH = 300 #Ajustement des textes à cette longueur

class ColumnDropper(BaseEstimator,TransformerMixin):

    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy().drop(self.columns, axis=1)

class LinksMaker(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy["links"] = "data/raw/images/image_train/image_" + X_copy.imageid.map(str) + "_product_" + X_copy.productid.map(str) + ".jpg"
        return X_copy
        
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

class WordsCounter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[f"words_in_"+column] = X_copy[column].apply(lambda text: get_num_words(text))
        return X_copy

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

class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, column, type_="tfidf"):
        self.column = column
        self.type_ = type_
        vectorizer_obj = TfidfVectorizer if self.type_=="tf" else CountVectorizer
        
        self.vectorizer =  vectorizer_obj(
                max_features=MAX_FEATURES_WORDS, 
                ngram_range=(1, 2),                 
                analyzer="word",
                preprocessor=clean_text,
                tokenizer=stemmatize_text,
                stop_words=french_stopwords
            )
        
    def fit(self, X, y=None):

        self.vectorizer.fit(X[self.column])

        return self    
        
    def transform(self, X):
        print(X.columns)
        X_copy = X.copy()
        X_vec = self.vectorizer.transform(X_copy[self.column]).toarray()

        df_vec = pd.DataFrame(
            X_vec,
            index=X_copy.index,
            columns=self.vectorizer.get_feature_names_out()
        )

        return X_copy.join(df_vec)#, rsuffix= "_" + self.type_)

def build_pipeline():

    pipeline = Pipeline(steps=[
        ("dropper", ColumnDropper(columns=["links", "productid", "imageid", "lang",])),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier() )
    ])

    return pipeline

def build_preprocessor():

    preprocessor = Pipeline(steps=[
        ('links', LinksMaker()),
        ('counters', WordsCounter(columns=["designation", "description"])),
        ('merger', TextColumnMerger(columns=["designation", "description"], name="text")),
        ('language', LanguageTransformer(column="text", text_length=500, translate=False)),
        #('vectorize_cn', Vectorizer(column="text", type_="cn")),
        ('vectorize_tf', Vectorizer(column="text", type_="tf")),
        ("dropper", ColumnDropper(columns=["designation", "description", "text"])),
        ])

    return preprocessor


def stemmatize_text(text):
    return [FrenchStemmer().stem(w) for w in word_tokenize(text)]

def clean_text(text):

    if isinstance(text, np.ndarray):
        text = text[0]
    
    #Suppression des balises
    text = re.sub('<[^<]+?>', ' ', text)

    #Suppression des autres choses que des lettres
    text = re.sub('[^A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02af\u1d00-\u1d25\u1d62-\u1d65\u1d6b-\u1d77\u1d79-\u1d9a\u1e00-\u1eff\u2090-\u2094\u2184-\u2184\u2488-\u2490\u271d-\u271d\u2c60-\u2c7c\u2c7e-\u2c7f\ua722-\ua76f\ua771-\ua787\ua78b-\ua78c\ua7fb-\ua7ff\ufb00-\ufb06]', ' ', text)

    #Suppression des maj
    text = text.lower()

    #Suppression des accents
    text = unidecode(text)

    #Ajustement de la taille des textes
    # if MAX_TEXT_LENGTH > 0:
    #     text = text[:MAX_TEXT_LENGTH]
    
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

# Lecture de fichier csv
@commons.timeit
def read_csv(name, folder):

    path = Path(folder, name)

    df =  pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    return df