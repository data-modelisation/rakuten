import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import os
import glob
import tensorflow as tf
import keras
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, recall_score,classification_report
from keras.applications.mobilenet_v2 import MobileNetV2
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
        load_embedding=False,
        
    ):
        self.load_embedding=load_embedding
        self.models_folder=models_folder
        self.suffix=suffix
        self.load=load
        self.batch_size=batch_size
        self.epochs=epochs

        print(f"model {type(self).__name__} init")
        return self
 
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
        """Set the callbacks for the model"""
        self.callbacks = [
                call_memory(),
                call_tqdm(),
                call_dashboard(path),
                call_checkpoint(path),
                call_earlystopping(),
                call_reducelr(),
            ] if self.is_neural else []

    def save_model(self, model, path):
        """Save the model"""
        if self.is_neural:
            model.save(path)
        else:
            joblib.dump(model, Path(path, "saved_model.pb"))
        print(f"model saved in {path}")


    def generate_path(self,):
        """Set the path"""
        path = Path(self.models_folder, self.model_type, self.model_name)
        path.mkdir(exist_ok=True, parents=True)
        
        return path

    def set_neural(self, model):
        """Check if the model is neural or not"""
        is_seq =  isinstance(model, keras.engine.sequential.Sequential)
        is_fon =  isinstance(model, keras.engine.functional.Functional)
        self.is_neural = is_seq or is_fon

    def load_model(self, path, ):
        """Load the model from path"""
        print(f"trying to load the model from {path}")
        try:
            model = tf.keras.models.load_model(path)
            print("model loaded")
            return model
        except Exception as exce:
            print(f"unable to find a model in the fodler {path}. Initialisation")
            return self.init_model()

    def get_model(self, path, train_dataset):
        """Load or initialize the model"""
        if self.load:
            return self.load_model(path)
        else:
            return self.init_model()

    def save_model_graph(self, model, path):
        """Save the model summary as a graph"""
        from keras.utils import plot_model
        
        plot_model(model, to_file=Path(path, 'model.png'))


    def fit(self, train_dataset, class_weight={}, validation=None):
        """Fit the model to the dataset"""

        self.set_neural(self.model)
        self.set_fit_kwargs(class_weight=class_weight, validation_data=validation)
        self.set_callbacks(self.path)

        self.model.compile(**self.compilation_kwargs)

        self.save_model_summary(self.model, self.path)
        self.save_model_graph(self.model, self.path)

        if not self.load:
            print(f"fitting the model {self.path}")
            history_model = self.model.fit(
                train_dataset, 
                callbacks=self.callbacks,
                **self.fit_kwargs
                )
            self.save_model(self.model, self.path)

            if self.model_type=="text":
                weights = tf.Variable(self.model.layers[0].get_weights()[0][1:])
                checkpoint = tf.train.Checkpoint(embedding=weights)
                checkpoint.save(self.embedding_layer_path)
                
        return self.model
    
    def graph_predictions_good_vs_bad(self, df_cross, path):
        """Graph with the good and the bad predictions as density"""
        values = df_cross.values
        mask_TP = np.eye(values.shape[0],dtype=bool)
        mask_OV = ~np.eye(values.shape[0],dtype=bool) 


        for name, color, mask in zip(("tp", "ov"), ("g", "r"), (mask_TP, mask_OV)):
            values_sns = values[mask]
           
            fig = sns.displot(x=values_sns,  kind="kde", color=color, fill=True, clip=(0,1))
            fig.set(xlim=(0, 1))
            plt.xlabel("Probabilités [-]")
            plt.ylabel("Densité [-]")
            plt.savefig(Path(path, f"density_{name}.svg"), format="svg")

    def start(self, ):
        """Load the model and get freeze the saved layers"""
        self.path = self.generate_path()
        
        self.model = self.load_model(self.path) if self.load else self.init_model()

        self.layers_folder_path = Path(self.path, "layers")
        self.layers_folder_path.mkdir(exist_ok=True, parents=True)
        self.embedding_layer_path = Path(self.layers_folder_path, "embedding.ckpt")
        
        if self.load_embedding:
            try:
                filename = glob.glob(str(self.layers_folder_path)+"/embedding*.index")[-1]
                self.embedding_layer_path = Path(self.layers_folder_path, Path(filename).stem)

                

                checkpoint = tf.train.Checkpoint(self.model)
                checkpoint.restore(self.embedding_layer_path)

                emb_layer = self.model.get_layer("te_emb")
                emb_layer.trainable = False
                print("embedding layer form disk, set as untrainable")
            except Exception as exce:
                print(f"unable to load the embedding layer : {exce}")
        else:
            print("embedding layer not initiazed, as asked")
        return self
        
  

    def predict(self, features, model=None, is_=None,for_api=False, enc_trues=None, generator=None):
        """Make a prediction on the features"""

        if (for_api is True) and (is_ == "text"):

            features_texts = np.array(features)
            lang_texts = np.array([get_lang(text) for text in features])
            translated_texts = np.array([translate_text(text, src=lang) for text, lang in zip(features, lang_texts)])
            cleaned_texts = np.array([clean_text(text) for text in translated_texts])
            encoded_texts = np.array([stemmatize_text(text) for text in cleaned_texts])
            dataset = tf.data.Dataset.from_tensor_slices((np.asarray(encoded_texts).astype(str), ))
            features = dataset.map(lambda x: generator.vectorize_text(x)).batch(1)
        
            
            vocab = generator.vectorize_layer.get_vocabulary()
            annotated = []
            for word in encoded_texts[0].split():
                if word in vocab:
                    annotated.append((word+" ", "V", "#d9e6f2"))
                else:
                    annotated.append(word+" ")
            
            
        elif (for_api is True) and (is_ == "image"):
            dataset = tf.data.Dataset.from_tensor_slices((np.asarray(features).astype(str), ))
            features = dataset.map(lambda x: generator.load_image(x)).batch(1)
                
        elif (for_api is True) and (is_ == "fusion"):
            def fusion_generator(texts, links, expand=False):
                    for text, link in zip(texts, links):
                        yield {"te_input": generator.vectorize_text(text, expand=expand), "im_input": generator.load_image(link)}
            
            
            texts = np.asarray([features[0],]).astype(np.str)
            links = np.asarray([features[1],]).astype(np.str)

            features_texts = np.array(texts)
            lang_texts = np.array([get_lang(text) for text in features])
            translated_texts = np.array([translate_text(text, src=lang) for text, lang in zip(features_texts, lang_texts)])
            cleaned_texts = np.array([clean_text(text) for text in translated_texts])
            encoded_texts = np.array([stemmatize_text(text) for text in cleaned_texts])

            vocab = generator.vectorize_layer.get_vocabulary()
            annotated = []
            for word in encoded_texts[0].split():
                if word in vocab:
                    annotated.append((word+" ", "V", "#d9e6f2"))
                else:
                    annotated.append(word+" ")

            features = tf.data.Dataset.from_generator(
                        fusion_generator,
                        args=[texts, links],
                        output_types = ({"te_input":tf.float32, "im_input":tf.float32})
                        ).batch(1)

        # if model is None:
        #     path = self.generate_path()
        #     self.model = self.load_model(path)
        # #model.compile(**self.compilation_kwargs)

        probas = self.model.predict(features)

        if (for_api is True) and (is_ == "image"):
            base_model = MobileNetV2(
                weights='imagenet', 
                include_top=False,
                input_shape = self.target_shape
            )
            print(base_model.summary())
            layers = [layer.output for layer in base_model.layers[:5]]
            
            activation_model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer("expanded_conv_project").output)
            activations = activation_model.predict(features)
            activation_layer = activations[0]
            
        enc_preds = np.argmax(probas, axis=1)
        rates = np.array([probas[idx, target] for idx, target in enumerate(enc_preds)])

        if generator is not None:
            dec_preds = generator.decode(enc_preds)
            nam_preds = generator.convert_to_readable_categories(pd.Series(dec_preds)).values
            
            enc_probas = np.arange(27)
            dec_probas = generator.decode(enc_probas)
            nam_probas = generator.convert_to_readable_categories(pd.Series(dec_probas)).values
            name_macro_probas = generator.convert_to_readable_macrocategories(pd.Series(dec_probas)).values
        else:
            dec_preds = [-1 for _ in enc_preds]
            nam_preds = ["na" for _ in enc_preds]

        if enc_trues is not None:
            dec_trues = generator.decode(enc_trues)
            nam_trues = generator.convert_to_readable_categories(pd.Series(dec_trues)).values

            df_cross = self.save_crosstab(dec_trues, dec_preds, self.path)
            self.graph_predictions_good_vs_bad(df_cross, self.path)
            report = self.save_classification_report(dec_trues, dec_preds, self.path)
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
                    "value probas": probas.tolist(),
                    "named probas": nam_probas.tolist(),
                    "macro named probas": name_macro_probas.tolist(),
                } 

            if is_ in ["text", "fusion"]:
                response["texts"] = features_texts.tolist()
                response["lang texts"] = lang_texts.tolist()
                response["translated texts"] = translated_texts.tolist()
                response["cleaned texts"] = cleaned_texts.tolist()
                response["encoded texts"] = encoded_texts.tolist()
                response["annotated texts"] = annotated
            if is_ in ["image"]:

                random_layer = np.random.randint(low=0, high=activation_layer.shape[2])
                response["activation_0"] = activation_layer[:,:,random_layer].tolist()
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
        """Calculate the crosstab between targets and predictions"""

        #Calculate the crosstab, normalisé selon les colonnes
        crosstab = pd.crosstab(
            y_true, y_pred,
            rownames=['Realité'], colnames=['Prédiction'],
            normalize="index"
        )

        #Add columns if missing (because unpredicted)
        mask_missing_columns = set(COLUMNS) - set(crosstab.columns)
        for miss_column in mask_missing_columns:
            crosstab.loc[:,miss_column] = np.zeros(len(COLUMNS))

        #Save it in a csv file
        crosstab.to_csv(Path(path, f'crosstab_report.csv'), index= True)

        return crosstab

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