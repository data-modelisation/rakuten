import numpy as np
import pandas as pd
import math
import tensorflow as tf
import os
import copy
from collections.abc import Iterable
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tools.text import pipeline_loader, pipeline_lang
from tools.commons import convert_to_readable_categories

class DataGenerator():
    def __init__(self,
        batch_size = 32,
        csv_data = "",
        csv_texts = "",
        csv_labels = "",
        root_dir="data/raw/images/image_train/",
        translate = True,
        stem=True,
        clean=True,
        samplings=None,
        random_state=123,
        exploration=False,
        from_data="raw",
        test_size=.1,
        valid_size=.1,
        target_shape=(),
        crop=False,
        from_api=False,
        embedding_dim=None,
        sequence_length=None,
        vocab_size=None,
        ):
        
        #Attributs
        self.vocab_size=vocab_size
        self.sequence_length=sequence_length
        self.embedding_dim=embedding_dim
        self.crop = crop
        self.clean=clean
        self.target_shape = target_shape
        self.test_size = test_size
        self.valid_size = valid_size
        self.from_data = from_data
        self.exploration = exploration
        self.batch_size = batch_size
        self.csv_texts = Path(csv_texts)
        self.csv_labels = Path(csv_labels)
        self.csv_data = Path(csv_data)
        self.root_dir = root_dir
        self.translate = translate
        self.stem = stem
        self.samplings = samplings
        self.random_state = random_state
        self.preprocessed = False
        self.encoder = None
        self.encoder_path = Path("./labelencoder.joblib")
        self.vectorize_layer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.sequence_length)

        if from_api:
            self.set_encoder()
        else:
            #Get or load the data (pd.DataFrame)
            self.data = self.load_data()

            #Encode the target (pd.DataFrame)
            self.encode()

            #Set the class_weights
            self.set_class_weight()

            print(f"{len(self.data)} observations and their targets loaded")

            if self.exploration:
                self.explore()

            self.data["names"] = convert_to_readable_categories(self.data.labels)


    #Conversion des numéros des categories en description
    def convert_to_readable_categories(self, serie:pd.Series):
        categories_num = [
            10, 40, 50, 60, 1140, 1160, 1180,
            1280, 1281, 1300, 1301, 1302, 1320, 1560,
            1920, 1940, 2060, 2220, 2280, 2403,
            2462, 2522, 2582, 2583, 2585, 2705, 2905
            ]

        categories_des = [
            "Livre occasion", "Jeu vidéo, accessoire tech.", "Accessoire Console", "Console de jeu", "Figurine", "Carte Collection", "Jeu Plateau",
            "Jouet enfant, déguisement", "Jeu de société", "Jouet tech", "Paire de chaussettes", "Jeu extérieur, vêtement", "Objet pour bébé", "Mobilier intérieur",
            "Chambre", "Cuisine", "Décoration intérieure", "Animal", "Revues et journaux", "Magazines, livres et BDs",
            "Jeu occasion", "Bureautique et papeterie", "Mobilier extérieur", "Autour de la piscine", "Bricolage", "Livre neuf", "Jeu PC"
            ]

        #categories_des_num = [f"{category_des}_{category_num}" for category_des, category_num in zip(categories_des, categories_num)]
        
        return serie.replace(
            to_replace = categories_num ,
            value = categories_des
        )

    def load_data(self,):

        if self.from_data=="raw":
            return self.read_raw()
        elif self.from_data == "csv":
            return self.read_csv()

    def set_class_weight(self,):
        values = self.data["targets"]
        classes = np.unique(values)
        weights = compute_class_weight('balanced', classes=classes, y=values)
        self.class_weight = {k: v for k, v in zip(classes, weights)}

        print(self.class_weight)

    def read_image_from_url(self, url):
        image_reader = tf.image.decode_jpeg(
            requests.get(url).content, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.int32)
        return float_caster

    def load_image(self, filepath):
   
        im = tf.io.read_file(filepath)
        im = tf.image.decode_jpeg(im, channels=3)
        if self.crop:
            im = im[50:450,50:450,:]
        im = tf.image.resize(im, size=self.target_shape[:2])
        return im

    def vectorize_text(self,text, expand=False):
        if expand:
            text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text)

    def build_datasets(self, dfs={}, bss={}):

        splits = ["train", "test", "valid"]
        types = ["text", "image", "fusion"]
        subdatasets = dict.fromkeys(types, None)
        datasets = dict.fromkeys(splits, subdatasets)
        datasets = {}
               

        for split in splits:
            datasets[split] = {}
            
            for type_ in types:

                print(f"building dataset for : {type_} | {split}")

                df = dfs.get(split)
                
                texts = df.text.astype(np.str)
                links = df.links.astype(np.str)
                targets = df.targets.values.astype(np.int32)


                def fusion_generator(texts, links, targets, expand=False):
                    for text, link, target in zip(texts, links, targets):
                        yield {"te_input": self.vectorize_text(text, expand=expand), "im_input": self.load_image(link)}, target

                if type_ == "text":

                    dataset = tf.data.Dataset.from_tensor_slices((texts, targets))
                    
                    if split == "train":
                        train_text_vectorizer = dataset.map(lambda x, y: x)
                        self.vectorize_layer.adapt(train_text_vectorizer.batch(64))

                        
                    dataset = dataset.map(lambda x,y : (self.vectorize_text(x), y))
                
                elif type_ == "image":
                    dataset = tf.data.Dataset.from_tensor_slices((links, targets))
                    dataset = dataset.map(lambda x,y : (self.load_image(x), y))
                
                elif type_ == "fusion":
                    dataset = tf.data.Dataset.from_generator(
                        fusion_generator,
                        args=[texts, links, targets],
                        output_types = ({"te_input":tf.float32, "im_input":tf.float32}, tf.int32)
                        )
               
                datasets[split][type_] = dataset.batch(bss.get(type_))
        
        return datasets

    def split(self,):

        train, test = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state)

        train, valid = train_test_split(
            train, 
            test_size=self.valid_size,
            random_state=self.random_state)

        return train, test, valid

    def sample(self, datasets):
        """Select a part of the dataset"""

        if self.samplings is not None:
            return [dataset.head(self.samplings) for dataset in datasets]
        else:
            return datasets

    def read_csv(self):
        #Read the file
        dataset = pd.read_csv(self.csv_data, index_col="Unnamed: 0")
        
        #Sample it
        dataset_sampled = self.sample([dataset,])[0]

        #Return the dataset
        return dataset_sampled

    def read_raw(self):

        #Read the csv files
        labels = pd.read_csv(self.csv_labels)
        texts = pd.read_csv(self.csv_texts)
        
        #Select a part of it
        labels, texts = self.sample([labels, texts])

        #Call the pipeline and transform the data (pd.DataFrame -> pd.DataFrame)
        text_loader = pipeline_loader(self.root_dir, clean=self.clean, stem=self.stem, translate=self.translate)
        data = text_loader.fit_transform(texts)
        
        #Add the targets to the dataframe        
        data["labels"] = labels.prdtypecode

        #If all the data is selected, we can save it in a new csv file
        if (self.samplings is None) or (self.samplings == len(data)):
            
            #The path contains the status of translation and stemming
            name = "data"+f"_cleaned_{int(self.clean)}"+f"_translated_{int(self.translate)}" + f"_stemmed_{int(self.stem)}.csv"
            
            #Save it
            data.to_csv(Path(self.csv_texts.parent, name))

        return data
   
    def set_encoder(self,):
        """Load or initialize the encoder"""
        
        #If there is no encoder
        if self.encoder is None:

            #We try to load it
            try:
                #Load the encoder
                self.encoder = joblib.load(self.encoder_path)
                print(f"load encoder from {self.encoder_path}")
                self.encoder_fitted = True
            except:
                #If that fails, we initialize a new encoer
                print("unable to load an existing encoder")
                self.encoder = LabelEncoder()
                print("brand new encoder created")
                self.encoder_fitted = False
                
        else:
            #Nothing to do, we arleady hae an encoder
            print("encoder already realdy")
            

    def encode(self):
        """Encode the targets with the LabelEncoder"""
        
        #Set the encoder
        self.set_encoder()
        
        if self.encoder_fitted:
            self.data["targets"] = self.encoder.transform(self.data.labels).astype(np.float32)
        else:
            self.data["targets"] = self.encoder.fit_transform(self.data.labels).astype(np.float32)
        
            #If the encoder was fitted, we save it
            if (self.samplings is None) or (self.samplings == len(self.data)):
                joblib.dump(self.encoder, self.encoder_path)

    def decode(self, values):
        """Decode the values with the LabelEncoder"""
        #Set the encoder
        #self.set_encoder()

        #From encoded values to original values
        return self.encoder.inverse_transform(values.astype(int))

    def explore(self):
        
        data_explore = self.data.copy()
        data_explore["prdtypename"] = convert_to_readable_categories(data_explore.labels)

        self.analysis_imbalanced(data_explore)
        self.analysis_words(data_explore)
        self.analysis_translation(data_explore)
        self.analysis_correlations(data_explore)

    def analysis_correlations(self, df):

        sns.set(rc={'figure.figsize': (5, 5)})

        for column in ["lang", "words_designation", "words_description"]:
            table = pd.crosstab(df[column], df.labels)
            table_norm = pd.crosstab(df[column], df.labels, normalize="columns")

            chi2, pvalue, dof, *_ = chi2_contingency(table)

            print("chi2 : ", chi2, " pvalue : ", pvalue)

            sns.heatmap(table_norm)
            plt.savefig(f"notebooks/images/chi_{column}.png")
            plt.clf()

        print("analysis imbalanced finished")

    def analysis_translation(self, df):

        sns.set(rc={'figure.figsize': (10, 5)})

        df_lang = df.lang.value_counts(normalize=True).sort_values(
            ascending=False).head(25).reset_index().rename({"index":"lang", "lang":"ratio"}, axis=1)

        graph_lang = sns.barplot(
            data=df_lang,
            x="lang",
            y="ratio",
        )
        graph_lang.set(
            xlabel='Langue [-]',
            ylabel='Nombre de titres / langue [-]',
            title='Analyse de la répartition des langues')

        plt.savefig("notebooks/images/lang.png")
        plt.clf()

        print("analysis language finished")

    def analysis_imbalanced(self, df):

        sns.set(rc={'figure.figsize': (10, 8)})

        graph_category = sns.countplot(
            data=df,
            y="prdtypename",
            order=df.prdtypename.value_counts().index)

        graph_category.set(
            xlabel='Nombre de produits [-]',
            ylabel='Catégories [-]',
            title='Nombre de produits par catégorie (Train)')
        plt.savefig("notebooks/images/imbalanced.png")
        plt.clf()

        print("analysis imbalanced finished")

    def analysis_words(self, df):

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        graph_desig = sns.histplot(
            data=df,
            kde=True,
            bins=40,
            x="words_designation",
            ax=axes[0])

        graph_descr = sns.histplot(
            data=df,
            kde=True,
            bins=40,
            x="words_description",
            ax=axes[1])

        graph_desig.set(
            xlabel='Nombre de mots [-]',
            ylabel='Total [-]',
            title='Nombre de mots dans `designation` (Train)')

        graph_descr.set(
            xlabel='Nombre de mots [-]',
            ylabel='Total [-]',
            title='Nombre de mots dans `description` (Train)')

        plt.savefig("notebooks/images/words.png")
        plt.clf()

        print("analysis words finished")
