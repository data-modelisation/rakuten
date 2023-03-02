import pandas as pd
import numpy as np
from pathlib import Path
from ftlangdetect import detect
import re
import logging
from unidecode import unidecode
from googletrans import Translator, LANGUAGES
#import swifter 
import fasttext
import pathlib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk import ngrams, FreqDist

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import commons

fasttext.FastText.eprint = lambda x: None

current_path = pathlib.Path().absolute()
nltk.data.path.append(current_path)
# nltk.download('stopwords', download_dir=current_path)
# nltk.download('punkt', download_dir=current_path)
# nltk.download('wordnet', download_dir=current_path)

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
    def __init__(self, path):
        self.path = path

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy["links"] = self.path + "/images/image_train/image_" + X_copy.imageid.map(str) + "_product_" + X_copy.productid.map(str) + ".jpg"
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

        lang = np.apply_along_axis(get_lang, 1, X[:,self.column].reshape(-1,1))

        translate_fct_vectorized = np.vectorize(translate_text)
        if self.translate:
            mask_not_fr = lang != "fr"
            X_copy[mask_not_fr, self.column] = translate_fct_vectorized(X_copy[:, self.column], src=lang, dest="fr")
        return lang

#Transformeur pour nettoyer le texte
class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, column, max_len=10):
        self.column = column
        self.max_len=max_len
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        clean_fct_vectorized = np.vectorize(clean_text)
        stemm_fct_vectorized = np.vectorize(stemmatize_text)

        X_copy[:, self.column] = clean_fct_vectorized(X_copy[:, self.column], self.max_len)
        #X_copy[:, self.column] = stemm_fct_vectorized(X_copy[:, self.column])

        return X_copy

#Transformeur pour vectoriser le texte
class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, 
        column, 
        vectorizer="tfidf",
        embedding=False,
        max_len=250,
        max_words_tokenized=100000,
        max_words_featured=43903
        ):

        self.column = column
        self.vectorizer = vectorizer
        self.max_len = max_len
        self.embedding = embedding

        max_words = max_words_tokenized if self.embedding else max_words_featured

        self.tokenizer =  Tokenizer(
            num_words=max_words,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
            char_level=False,
            oov_token=None,
            analyzer=None,
        )
        
    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X[:, self.column])

        print("The document count", self.tokenizer.document_count)

        return self    
        
    def transform(self, X):

        sequences = self.tokenizer.texts_to_sequences(X[:, self.column])

        if not self.embedding:
            return self.tokenizer.sequences_to_matrix(sequences, mode=self.vectorizer)
        else:
            return pad_sequences(sequences,
                padding="post",
                truncating='post',
                maxlen=self.max_len)
        


# Construction du pipeline pour le chargement des données
def pipeline_loader():
    loader = Pipeline(steps=[
        ('counter', TextCounter(columns=["designation", "description"])),
        ('merger', TextColumnMerger(columns=["designation", "description"], name="text")),
        ("dropper", ColumnDropper(columns=["Unnamed: 0", "designation", "description", "imageid", "productid"])),
    ])
    return loader

# Construction du pipeline pour la traduction des langues
def pipeline_lang(translate=False):

    translater = Pipeline(steps=[
        ('trans', LanguageTransformer(column=4, text_length=500, translate=translate)),
    ])
    return translater

# Construction du pipeline pour le preprocessing
def pipeline_preprocess(vectorizer="", embedding=False, max_words_featured=2000, max_words_tokenized=100, max_len=100):

    preprocessor = Pipeline(steps=[        
        ('cleaner', TextCleaner(column=4, max_len=max_len)),
        ('vectorizer', Vectorizer(column=4, vectorizer=vectorizer, embedding=embedding, max_words_featured=max_words_featured, max_words_tokenized=max_words_tokenized, max_len=max_len)),
        ])

    # if not embedding:
    #     preprocessor.steps.append(['dropper',ColumnDropper(columns=[4, 5])])

    return preprocessor

#Stemmatisation d'un texte en francais
def stemmatize_text(text):
    return [FrenchStemmer().stem(w) for w in word_tokenize(text)]

#Nettoyage des textes
def clean_text(text, max_words):
   
    #Suppression des balises
    text = re.sub('<[^<]+?>', ' ', text)

    #Suppression des autres choses que des lettres
    text = re.sub('[^A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02af\u1d00-\u1d25\u1d62-\u1d65\u1d6b-\u1d77\u1d79-\u1d9a\u1e00-\u1eff\u2090-\u2094\u2184-\u2184\u2488-\u2490\u271d-\u271d\u2c60-\u2c7c\u2c7e-\u2c7f\ua722-\ua76f\ua771-\ua787\ua78b-\ua78c\ua7fb-\ua7ff\ufb00-\ufb06]', ' ', text)

    #Suppression des maj
    text = text.lower()

    #Suppression des accents
    text = unidecode(text)

    #Ajustement de la taille des textes
    if max_words > 0:
        text = text[:int(max_words*10)]

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
def get_lang(text, text_length=300):

    if isinstance(text, np.ndarray):
        text = text[0]

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