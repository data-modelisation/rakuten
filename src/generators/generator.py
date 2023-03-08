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

from src.tools.text import pipeline_loader
from src.tools.commons import convert_to_readable_categories

class DataGenerator():
    def __init__(self,
        batch_size = 32,
        csv_data = "",
        csv_texts = "",
        csv_labels = "",
        root_dir="data/raw/images/image_train/",
        translate = True,
        stem=True,
        samplings=None,
        random_state=123,
        exploration=False,
        ):
        
        #Attributs
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
        self.encoder_path = Path("src/generators/labelencoder.joblib")
        
        #Get or load the data (pd.DataFrame)
        self.data = self.load() if str(self.csv_data) != "." else self.read()
        
        #Encode the target (pd.DataFrame)
        self.encode()

        #Convert in to floats
        self.make_float()

        #Set the class_weights
        self.set_class_weight()

        print(f"{len(self.data)} observations and their targets loaded")

        if self.exploration:
            self.explore()


    def make_float(self):
        columns_number = self.data.select_dtypes(np.number).columns
        self.data[columns_number] = self.data[columns_number].astype(np.float32)

    def set_class_weight(self,):
        values = self.data["targets"]
        classes = np.unique(values)
        weights = compute_class_weight('balanced', classes=classes, y=values)
        self.class_weight = {k: v for k, v in zip(classes, weights)}

    def set_preprocessed(self, value):
        self.preprocessed = value

    def preprocess(self, preprocessor, type_="text"):

        print(f"preprocess for type {type_}")
        if type_ == "text" or type_ == "fusion":
            if preprocessor is not None:
                data_transformed = preprocessor.transform(self.data.values)
            else:
                data_transformed = self.data.values[:, :-1]
        elif type_ == "image":
            data_transformed = self.data.values[:, 0].reshape(-1,1)
        
        # if type_ == "fusion":
        #     import pdb; pdb.set_trace()
        #     data_transformed = np.concatenate([self.data.values[:, 0].reshape(-1,1), data_transformed], axis=1)

        self.data_transformed = np.concatenate([data_transformed, self.data.targets.values.reshape(-1,1)], axis=1)
        
        self.preprocessed=True

        print(f"data transformed with preprocessor as NumpyArray : shape {self.data_transformed.shape}")
        return self

    def load_image(self, x):
   
        filepath = x[0]
        im = tf.io.read_file(filepath)
        im = tf.image.decode_jpeg(im, channels=3)
        im = tf.image.resize(im, size=(50, 50, 3))
        return im

    def split(self, indexes=None, test_size=.16):

        if indexes is None:
            indexes = train_test_split(np.arange(len(self.data)), test_size=test_size, random_state=self.random_state)

        generators = []
        for index in  indexes:
            generator = copy.deepcopy(self)

            if isinstance(self.data, pd.core.frame.DataFrame):
                generator.data = generator.data.loc[index].reset_index().drop("index", axis=1)
            else:
                generator.data = generator.data[index,:]
            
            generators.append(generator)
        
        return generators

    def sample(self, datasets):
        """Select a part of the dataset"""

        if self.samplings is not None:
            return [dataset.head(self.samplings) for dataset in datasets]
        else:
            return datasets

    def load(self):
        #Read the file
        dataset = pd.read_csv(self.csv_data, index_col="Unnamed: 0")
        
        #Sample it
        dataset_sampled = self.sample([dataset,])[0]

        #Return the dataset
        return dataset_sampled

    def read(self):
        #Read the csv files
        labels = pd.read_csv(self.csv_labels)
        texts = pd.read_csv(self.csv_texts)
        
        #Select a part of it
        labels, texts = self.sample([labels, texts])

        #Call the pipeline and transform the data
        text_loader = pipeline_loader(self.root_dir, translate=self.translate, stem=self.stem)
        data = text_loader.fit_transform(texts)

        #Add the targets to the dataframe        
        data["labels"] = labels.prdtypecode

        #If all the data is selected, we can save it in a new csv file
        if (self.samplings is None) or (self.samplings == len(data)):
            
            #The path contains the status of translation and stemming
            name = "data"+f"_translated_{int(self.translate)}" + f"_stemmed_{int(self.stem)}.csv"
            
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
        print(self.encoder)
        
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
        
        self.data_explore = self.data.copy()
        self.data_explore["prdtypename"] = convert_to_readable_categories(self.data_explore.labels)

        self.analysis_imbalanced()
        self.analysis_words()
        self.analysis_translation()
        self.analysis_correlations()

    def analysis_correlations(self,):

        sns.set(rc={'figure.figsize': (5, 5)})

        for column in ["lang", "words_designation", "words_description"]:
            table = pd.crosstab(self.data_explore[column], self.data_explore.labels)
            table_norm = pd.crosstab(self.data_explore[column], self.data_explore.labels, normalize="columns")

            chi2, pvalue, dof, *_ = chi2_contingency(table)

            print("chi2 : ", chi2, " pvalue : ", pvalue)

            sns.heatmap(table_norm)
            plt.savefig(f"notebooks/images/chi_{column}.png")
            plt.clf()

        print("analysis imbalanced finished")

    def analysis_translation(self,):

        sns.set(rc={'figure.figsize': (10, 5)})

        order_lang = self.data_explore.lang.value_counts(normalize=True).sort_values(
            ascending=False).head(25).reset_index().rename({"index":"lang", "lang":"ratio"}, axis=1)

        graph_lang = sns.barplot(
            data=order_lang,
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

    def analysis_imbalanced(self,):

        sns.set(rc={'figure.figsize': (10, 8)})

        graph_category = sns.countplot(
            data=self.data_explore,
            y="prdtypename",
            order=self.data_explore.prdtypename.value_counts().index)

        graph_category.set(
            xlabel='Nombre de produits [-]',
            ylabel='Catégories [-]',
            title='Nombre de produits par catégorie (Train)')
        plt.savefig("notebooks/images/imbalanced.png")
        plt.clf()

        print("analysis imbalanced finished")

    def analysis_words(self,):

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        graph_desig = sns.histplot(
            data=self.data_explore,
            kde=True,
            bins=40,
            x="words_designation",
            ax=axes[0])

        graph_descr = sns.histplot(
            data=self.data_explore,
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

# class CommonGenerator(tf.keras.utils.Sequence):

#     def __init__(self,
#         *args,
#         **kwargs,
#         ):

#         self.batch_size = kwargs.get("batch_size")
#         self.csv_texts = kwargs.get("csv_texts")
#         self.csv_labels = kwargs.get("csv_labels")
#         self.preprocessor = kwargs.get("preprocessor")
#         self.preprocessor_fit = kwargs.get("preprocessor_fit")
#         self.encoder = kwargs.get("encoder")
#         self.encoder_fitted = kwargs.get("encoder_fitted")
#         self.encoder_params = kwargs.get("encoder_params")

#     def fit_preprocess(self,):
#         if self.preprocessor_fit:
#             self.features = self.preprocessor.fit_transform(self.features)
#         else:
#             self.features = self.preprocessor.transform(self.features)

#     def encode_targets(self,):
#         if self.encoder is None:
#             self.encoder = LabelEncoder()
            
#         if not self.encoder_fitted:
#             self.data["targets"] = self.encoder.fit_transform(self.labels)
#             self.encoder.get_params()

#         else:
#             self.data["targets"] = self.encoder.transform(self.labels)

#         self.__set_class_weigth__()

#     def __splitter__(self, split_indexes, type_=None, is_batch=True):
        
#         splitted_generator = copy.deepcopy(self)
#         if is_batch:
#             indexes = list(self.flatten([self.__get_batch_indexes__(idx_1D) for idx_1D in split_indexes]))
#         else:
#             indexes = split_indexes

#         splitted_generator.targets = splitted_generator.targets[indexes]
#         splitted_generator.features = splitted_generator.features[indexes]

#         return splitted_generator

#     def split(self, split_indexes=[], types=["train", "test"], is_batch=True):
        
#         return [self.__splitter__(indexes, is_batch=is_batch, type_=type_) for indexes, type_ in zip(split_indexes, types)]

#     def __set_class_weigth__(self,):
#         values = self.targets
#         classes = np.unique(values)
#         weights = compute_class_weight('balanced', classes=classes, y=values)
#         self.class_weight = {k: v for k, v in zip(classes, weights)}

#     def __len__(self):
#         return math.ceil(len(self.targets) / self.batch_size)

#     def __get_batch_indexes__(self, idx):
#         max_idx_possible = min(len(self.targets), (idx + 1) * self.batch_size)
#         return np.arange(start=idx * self.batch_size, stop=max_idx_possible, dtype=int)

#     def flatten(self, xs):
        
#         for x in xs:
#             if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
#                 yield from self.flatten(x)
#             else:
#                 yield x

#     def decode(self, values):
#         return self.encoder.inverse_transform(values)

#     def plot_bits(self):

#         bits = np.array(255.0 / (self.targets + 1), dtype=np.int)
#         num_pixels = int(np.ceil(np.sqrt(len(self.targets))))

#         R = np.empty((num_pixels*num_pixels,))
#         R[np.arange(len(self.targets), dtype=int)] = bits
#         plt.imshow(R.reshape(num_pixels,num_pixels, 1), cmap="gray")
#         plt.savefig("notebooks/images/random_targets.png")


#     def __getitem__(self, batch_idx):

#         indexes = self.__get_batch_indexes__(batch_idx)

#         features = np.array(self.features[indexes])
#         targets = np.array(self.targets[indexes])

#         return features, targets

    
    
