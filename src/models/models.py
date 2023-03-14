import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import os
import tensorflow as tf
import keras

from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, recall_score,classification_report

from imblearn.metrics import classification_report_imbalanced

from models.models_utils import call_memory, call_tqdm, call_dashboard, call_checkpoint, call_earlystopping, call_reducelr
#from src.generators.generator_fusion import FusionGenerator
#from src.generators.generator_image import ImageGenerator
from models.models_utils import METRICS
from tools.text import get_lang, translate_text, stemmatize_text, clean_text

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RNN, GRUCell
from tensorflow.keras.layers import Embedding, Rescaling, TextVectorization
from tensorflow.keras.models import Model

COLUMNS = [10,40,50,60,1140,1160,1180,1280,1281,1300,1301,1302,1320,1560,1920,1940,2060,2220,2280,2403,2462,2522,2582,2583,2585,2705,2905,]

class MyDataSetModel():
    def __init__(self,
        models_folder="models/",
        suffix="",
        load=False,
        batch_size=30,
        epochs=50,
        
    ):
        self.models_folder=models_folder
        self.suffix=suffix
        self.load=load
        self.batch_size=batch_size
        self.epochs=epochs

    @property
    def model_name(self,):
        """Extract the model name from it class name"""
        return type(self).__name__.lower().replace("model", "") + self.suffix

    @property
    def model_type(self,):
        """Extract the model type from its name"""
        if "text" in self.model_name:
            return "text"
        elif "image" in self.model_name:
            return "image"
        elif "fusion" in self.model_name:
            return "fusion"

    @property
    def compilation_kwargs(self):
        """Configuration parameters for neural networks"""

        return {
            "optimizer" : tf.keras.optimizers.Adam(),
            "loss" : tf.keras.losses.SparseCategoricalCrossentropy(),
            "metrics" : [tf.keras.metrics.SparseCategoricalAccuracy()]
        }

    def set_fit_kwargs(self, class_weight={}, validation_data=None):
        """Dictionnaire des modèles"""

        self.fit_kwargs = {
                "batch_size":self.batch_size,
                "epochs" : self.epochs,
                "validation_data" : validation_data,
                "verbose":0,
                #"class_weight" : class_weight,
            } if self.is_neural else {}

    def set_callbacks(self, path):
        self.callbacks = [
                call_memory(),
                call_tqdm(),
                call_dashboard(path),
                call_checkpoint(path),
                call_earlystopping(),
                call_reducelr(),
            ] if self.is_neural else []

    def save_model(self, model, path):
        if self.is_neural:
            model.save(path)
        else:
            joblib.dump(model, Path(path, "saved_model.pb"))
        print(f"model saved in {path}")


    def generate_path(self,):

        path = Path(self.models_folder, self.model_type, self.model_name)
        path.mkdir(exist_ok=True, parents=True)
        
        return path

    def set_neural(self, model):
        is_seq =  isinstance(model, keras.engine.sequential.Sequential)
        is_fon =  isinstance(model, keras.engine.functional.Functional)
        self.is_neural = is_seq or is_fon

    def load_model(self, path, dataset):

        try:
            return tf.keras.models.load_model(path)
        except Exception as exce:
            print(f"unable to find a model in the fodler {path}")
            return self.init_model(dataset)

    def get_model(self, path, train_dataset):

        if self.load:
            return self.load_model(path, train_dataset)
        else:
            return self.init_model(train_dataset)

    def save_model_graph(self, model, path):
        from keras.utils import plot_model
        
        plot_model(model, to_file=Path(path, 'model.png'))


    def fit(self, train_dataset, class_weight={}, validation=None):
        
        path = self.generate_path()
        model = self.get_model(path, train_dataset)
        self.set_neural(model)
        self.set_fit_kwargs(class_weight=class_weight, validation_data=validation)
        self.set_callbacks(path)

        model.compile(**self.compilation_kwargs)

        self.save_model_summary(model, path)
        self.save_model_graph(model, path)

        if not self.load:
            print(f"fitting the model {path}")
            history_model = model.fit(
                train_dataset, 
                callbacks=self.callbacks,
                **self.fit_kwargs
                )
            self.save_model(model, path)
            

        return model
    
    def predict(self, features, model=None, is_=None,for_api=False, enc_trues=None, generator=None):
        

        if (for_api is True) and (is_ == "text"):

            features_texts = np.array(features)
            lang_texts = np.array([get_lang(text) for text in features])
            translated_texts = np.array([translate_text(text, src=lang) for text, lang in zip(features, lang_texts)])
            cleaned_texts = np.array([clean_text(text) for text in translated_texts])
            encoded_texts = np.array([stemmatize_text(text) for text in cleaned_texts])
            dataset = tf.data.Dataset.from_tensor_slices((np.asarray(features).astype(str), ))
            features = dataset.map(lambda x: generator.vectorize_text(x)).batch(1)
            print("features",features)
        
        elif (for_api is True) and (is_ == "image"):
            dataset = tf.data.Dataset.from_tensor_slices((np.asarray(features).astype(str), ))
            features = dataset.map(lambda x: generator.load_image(x)).batch(1)
                
        path = self.generate_path()
        model = self.load_model(path, features)
        model.compile(**self.compilation_kwargs)

        probas = model.predict(features)

        enc_preds = np.argmax(probas, axis=1)
        rates = np.array([probas[idx, target] for idx, target in enumerate(enc_preds)])

        if generator is not None:
            dec_preds = generator.decode(enc_preds)
            nam_preds = generator.convert_to_readable_categories(pd.Series(dec_preds)).values
            
            enc_probas = np.arange(27)
            dec_probas = generator.decode(enc_probas)
            nam_probas = generator.convert_to_readable_categories(pd.Series(dec_probas)).values

        else:
            dec_preds = [-1 for _ in enc_preds]
            nam_preds = ["na" for _ in enc_preds]

        if enc_trues is not None:
            dec_trues = generator.decode(enc_trues)
            nam_trues = generator.convert_to_readable_categories(pd.Series(dec_trues)).values
            print(path)
            self.save_crosstab(dec_trues, dec_preds, path)
            report = self.save_classification_report(dec_trues, dec_preds, path)
            self.push_classification_to_summary(report)
        else:
            enc_trues = np.array([-1 for _ in enc_preds])
            dec_trues = np.array([-1 for _ in enc_preds])
            nam_trues = np.array(["na" for _ in enc_preds])

            
        if for_api:
            response =  {       
                    "encoded predictions": enc_preds.tolist(),
                    "encoded trues" : enc_trues.tolist(),
                    "confidences": rates.tolist(),
                    "decoded predictions":dec_preds.tolist(),
                    "decoded trues" : dec_trues.tolist(),
                    "named predictions":nam_preds.tolist(),
                    "named trues" : nam_trues.tolist(),
                    "value probas":probas.tolist(),
                    "named probas":nam_probas.tolist(),
                } 

            if is_ == "text":
                response["texts"] = features_texts.tolist()
                response["lang texts"] = lang_texts.tolist()
                response["translated texts"] = translated_texts.tolist()
                response["cleaned texts"] = cleaned_texts.tolist()
                response["encoded texts"] = encoded_texts.tolist()
                

            return response

    def save_model_summary(self, model, path):
        """Save the description of the layers of the model into a file"""

        #If it's a neural network
        if self.is_neural:
            path = Path(path, "model_summary.txt")

            #Save the model summary (layers) to a txt file
            with open(path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
                
    def save_crosstab(self, y_true, y_pred, path):
        """"Calculate the crosstab between targets and predictions"""

        #Calculate the crosstab, normalisé selon les colonnes
        crosstab = pd.crosstab(
            y_true, y_pred,
            rownames=['Realité'], colnames=['Prédiction'],
            normalize="index"
        )
        #Save it in a csv file
        crosstab.to_csv(Path(path, f'crosstab_report.csv'), index= True)

    def save_classification_report(self, y_true, y_pred, path):
        """Create and save a classification report"""

        #Create clf report
        clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        #Create a df with the report
        clf_report = pd.DataFrame(clf_report).transpose()

        #Save the dataframe as csv file
        #if self.save:
        clf_report.to_csv(Path(path, f'clf_report.csv'), index= True)
        
        #Return the clf report
        return clf_report

    def push_classification_to_summary(self, report, metrics=["precision", "recall", "f1-score"]):

        #For all metrics
        for metric in metrics:

            #Create a path to the summary file
            summary_path = Path(self.models_folder, self.model_type, f"summary_{metric}.csv")
            
            #Get the columns (targets) and the values (scores per category)
            columns = report[metric].index.tolist()
            values = report[metric].values

            #Expand the values : it's possible that a class does not exist in the predicted values
            expected_columns = COLUMNS + ["accuracy", "macro avg", "weighted avg"]
            expected_values = []   
            for column in expected_columns:
                if str(column) in columns:
                    expected_values.append(report[metric].loc[str(column)])
                else:
                    expected_values.append(0.0)

            # Read the existing dataframe or create it
            df = pd.read_csv(summary_path, delimiter=",", index_col=0) if summary_path.exists() else pd.DataFrame(index=expected_columns)

            #Create or replace the scores obtained by the model
            df[self.model_name] = expected_values
            
            #Save it back to the summary file
            df.to_csv(summary_path)