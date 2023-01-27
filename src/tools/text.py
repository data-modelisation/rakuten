import pandas as pd
from pathlib import Path
from ftlangdetect import detect
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk import ngrams, FreqDist
from . import commons

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Selection des 500 mots les plus communs
NUM_COMMON_WORDS = 500

# Feature-engineering du texte
@commons.timeit
def apply_feature_engineering(df:pd.DataFrame):
    logging.debug("start feature-engineering")

    #Remplacement de Nan par ""
    df["description"] = df.description.fillna("")
    logging.debug("nans converted to empty string")

    #Nombre de mots dans designation
    df["desi_num_words"] = df.designation.apply(lambda x: len(get_text_words(x)))
    logging.debug(f"mean num words in designation : {df.desi_num_words.mean()}")

    #Nombre de mots dans designation
    df["desc_num_words"] = df.description.apply(lambda x: len(get_text_words(x)))
    logging.debug(f"mean num words in designation : {df.desc_num_words.mean()}")

    #Fusion description + designation
    df["text"] = commons.merge_columns(df.designation, df.description)
    logging.debug(f"merging of columns")

    #Suppression de la ponctuation
    df["text_clean"] = remove_balise(df.text)
    logging.debug(f"balise removed")

    #Langue utilisée
    df["lang"] = find_language(df.text_clean)
    logging.debug(f"got the langages")

    #Suppression de la ponctuation
    df["text_clean"] = remove_punctuation(df.text_clean)
    logging.debug(f"punctuation removed")

    #Suppression des Maj
    df["text_clean"] = make_lowercase(df.text_clean)
    logging.debug(f"text lowered")

    #Suppression des nombres
    df["text_clean"] = remove_numerical(df.text_clean)
    logging.debug(f"numbers removed")

    #Nombre de mots dans text
    df["num_words"] = df.text_clean.apply(lambda x: len(get_text_words(x)))
    logging.debug(f"mean num words in text : {df.num_words.mean()}")

    #Suppression des stopwords
    df["text_clean"] = df.apply(lambda x: remove_stops(x.text_clean, x.lang), axis=1)
    logging.debug(f"stopwords removed")

    # #Suppression des mots inutiles
    # df["text_clean"] = df.text_clean.apply(lambda x: remove_words(x))
    # logging.debug(f"useless words removed")

    #Passage à la racine des mots
    df["text_stem"] = stemmatize_serie(df)
    logging.debug(f"text stemmed")

    #Selection des mots les plus courant et chaque mot courant a une colonne 
    df_dummy, df_commons = select_common_words(df.text_stem)
    
    df = pd.concat([df, df_dummy], axis=1)

    return df, df_commons

@commons.timeit
def select_common_words(serie:pd.Series, ngram_size:int=1):
    
    #On récupère tous les textes nettoyés
    all_texts = []
    for text in serie.values:
        all_texts += text
    
    # On cherche les n-gramms
    all_counts = FreqDist(ngrams(all_texts, ngram_size))
    
    
    most_common_words = {
        "words" : [],
        "values" : [],
    }
    #On recupere les NUM_COMMON_WORDS mots les plus souvent utilisés
    most_common_words["words"] = [item[0][0] for item in all_counts.most_common(NUM_COMMON_WORDS)]
    most_common_words["values"] = [item[1] for item in all_counts.most_common(NUM_COMMON_WORDS)]


    df_common = pd.DataFrame.from_dict(most_common_words, orient="columns")
    #On fait 
    table_words_used = serie.apply(lambda x: [word for word in x if word in most_common_words])

    table_common_words = [[common_word in words for common_word in most_common_words.get("words")] for words in table_words_used.values]
    
    #On créé un dataframe avec table_common_words comme valeurs
    df = pd.DataFrame(
        table_common_words,
        columns=most_common_words.get("words"),
        index = serie.index
    )

    return df, df_common


def stemmatize_words(words:list, lang:str):

    if lang == "fr":
        stemmed = [FrenchStemmer().stem(word) for word in words]

    elif lang == "en":
        stemmed = [PorterStemmer().stem(word) for word in words]

    else:
        stemmed = words

    #On renvoie la liste des mots sans duplicats
    return list(set(stemmed))

@commons.timeit
def stemmatize_serie(df):
    return df.apply(lambda x: stemmatize_words(x.text_clean, x.lang), axis=1)

#Suppression des stopwords et text -> word
def remove_stops(text:str, lang):

    langage = {
        "en": "english",
        'fr':"french",
        "de":"german",
        'it': "italian"
        }

    if lang in langage.keys():
        stopWords = set(stopwords.words(langage.get(lang)))
        
        words = word_tokenize(text)
        wordsFiltered = []

        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)

        return wordsFiltered
    else:
        return []

#Trouver les racines des mots
def lemmatize_text(words, lemmatizer=WordNetLemmatizer()):
    return [lemmatizer.lemmatize(w) for w in words]

#Suppression de la ponctuation (sauf ')
@commons.timeit
def remove_punctuation(serie:pd.Series):
    return serie.apply(lambda text: re.sub(r"[^\w\s']", ' ', text))

#Suppression des balises
@commons.timeit
def remove_balise(serie:pd.Series):
    return serie.apply(lambda text : re.sub('<[^<]+?>', '', text))

#Suppression des nombres
@commons.timeit
def remove_numerical(serie:pd.Series):
    return serie.apply(lambda text : re.sub(r'[0-9]+', '', text))

#Suppression des mots courts
def remove_short_words(words:list, length_min:int=2):
    return [word for word in words if len(word) >=  length_min]

#Passage en minuscule
@commons.timeit
def make_lowercase(serie:pd.Series):
    return serie.apply(lambda text : text.lower())

#Langue d'un texte
def get_lang(text:str):
    
    max_length = min(200, len(text), ) #Limitation de longueur

    try:
        return detect(text=text[:max_length], low_memory=True).get("lang")
    except Exception as exce:
        return "unkown"

# Detection de langue
@commons.timeit
def find_language(serie:pd.Series):
    return serie.apply(lambda text : get_lang(text))

# Longueur d'un texte
def get_text_length(text:str):
    try:
        return len(text)
    except Exception as exce:
        return 0

# Nombre de mots d'un texte
def get_text_words(text:str):
    try:
        return text.split()
    except Exception as exce:
        return []

# Lecture de fichier csv
@commons.timeit
def read_csv(name, folder):

    path = Path(folder, name)

    df =  pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    return df