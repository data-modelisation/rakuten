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

from src.models.models_utils import call_memory, call_tqdm, call_dashboard, call_checkpoint, call_earlystopping, call_reducelr
from src.generators.generator_fusion import FusionGenerator
from src.generators.generator_image import ImageGenerator

COLUMNS = [10,40,50,60,1140,1160,1180,1280,1281,1300,1301,1302,1320,1560,1920,1940,2060,2220,2280,2403,2462,2522,2582,2583,2585,2705,2905,]

class Model():

    def __init__(self, 
        generator = None,
        generator_test = None,
        num_folds=3, 
        epochs=2,
        batch_size=10,
        load=True,   
        save=True,
        report=True,
        summary=True,
        predict=False,
        folder="src/models/",
        name=None,
        suffix="",
        random_state=42,
        ):

        self.preprocessor = None
        self.preprocessor_fitted = False
        self.model_fitted = False
        self.model = None
        self.suffix=suffix
        self.random_state=random_state
        
        self.generator = generator
        self.generator_test = generator_test
        
        self.num_folds = num_folds
        self.models_folder = folder
        self.batch_size = batch_size
        self.epochs = epochs
        self.summary = summary
        self.report=report
        self.load = load
        self.save = save
        self.model_path = None
        self.model_current_path = None

        return None

    def set_model_path(self,):
        self.model_path = Path(self.models_folder, self.type, self.get_name())

    def get_name(self):
        return self.name + self.suffix

    def fit_kwargs(self, validation_data=None):

        return {
                "batch_size":self.batch_size,
                "epochs" : self.epochs,
                "validation_data" : validation_data,
                "verbose":0,
                "class_weight" : self.class_weight,
                "callbacks" : self.callbacks
            } if self.model_neural else {}
            
    @property
    def has_preprocessor(self):
        return self.get_preprocessor() is not None

    @property
    def callbacks(self):
        return [
                call_memory(),
                call_tqdm(),
                call_dashboard(self.model_current_path),
                call_checkpoint(self.model_current_path),
                call_earlystopping(),
                call_reducelr(),
            ] if self.model_neural else []

    

    def load_preprocess(self):
        if self.model_path is None:
            self.set_model_folder()

        if self.has_preprocessor:
            self.preprocessor = joblib.load(Path(self.model_path, 'preprocess.joblib'))
            
        self.preprocessor_fitted = True


    def save_preprocess(self):
        if self.preprocessor_fitted and self.save:
            path = Path(self.model_path, 'preprocess.joblib')
            joblib.dump(self.preprocessor, path)
            print(f"preprocess savec in {path}")
            
    def load_model(self, fold=None):
        
        self.set_current_folder(fold=fold)

        if self.model_neural:
            model = tf.keras.models.load_model(self.model_current_path)#(Path(path, "saved_model.hdf5"))
        else:
            model = joblib.load(Path(self.model_current_path, "saved_model.pb"))
        
        print(f"model loaded from {self.model_current_path}")
        return model

    def save_model(self, model, in_fold=False):
        if self.model_neural:
            model.save(self.model_current_path)
        else:
            joblib.dump(model, Path(self.model_current_path, "saved_model.pb"))
        print(f"model saved in {self.model_current_path}")

    def fit_preprocessor(self, values):

        if not self.preprocessor:
            self.preprocessor = self.get_preprocessor()

        values_transformed = self.preprocessor.fit_transform(values) if self.preprocessor else values

        self.preprocessor_fitted = True
        if self.preprocessor:
            self.save_preprocess()

        return values_transformed

    def get_model(self, fold=None):
        if self.load:                
            return self.load_model(fold=fold)
        else:
            return self.init_model()


    def unpack_iterator(self, iterator):
        *unpacked, = iterator
        x = []
        y = []
        for array in unpacked:
            x.append(array[0])
            y.append(array[1])
        return np.concatenate(x), np.concatenate(y)

    def set_current_folder(self, fold=None):
        
        if self.model_path is None:
            self.set_model_path()


        if fold is None or fold < -1:
            self.model_current_path = self.model_path
        else:
            self.model_current_path = Path(self.model_path, "folds", f"fold_{fold:02d}")

        self.model_current_path.mkdir(exist_ok=True, parents=True)
 

    def flow_if_necessary(self, input, type="validation"):
        
        return input.flow(type_=type) if isinstance(input, ImageGenerator) else input


    def fit(self, model=None, train_data=None, validation_data=None, fold=None, crossval=True, class_weight=None):
        
        #Set the weights
        self.class_weight = class_weight if class_weight is not None else train_data.class_weight 

        #If it's a fusion model, we build a special sequence with texts and images
        if self.type == "fusion" and not isinstance(train_data, FusionGenerator):
            train_data = FusionGenerator(train_data)
            validation_data = FusionGenerator(validation_data)

            print("data transformed to sequences for fusion")
            fold = 1
      
        if crossval:
            scores = []

            for kfold, (train_index, valid_index) in enumerate(KFold(self.num_folds).split(train_data)):

                kfold_train_data = train_data 
                kfold_valid_data = validation_data 
                
                # Entrainement sur les donne entrainement
                model = self.fit( 
                    train_data=kfold_train_data, 
                    validation_data=kfold_valid_data, 
                    fold=kfold,
                    crossval=False, 
                    class_weight=class_weight)

                #Make a prediciton
                kscore = self.predict(model, 
                    validation_data,
                    validation_data.targets,
                    train_data)
                
                scores.append(kscore)
            
            #Copy the best to the model root folder
            model = self.select_best_fold(
                scores
            )

            self.set_current_folder(fold=None)
            self.save_model(model)

            #Make a prediction
            self.predict(model, 
                validation_data,
                validation_data.targets,
                train_data)
        
        else:

            #Set the folder
            self.set_current_folder(fold=fold)
            print(f"Current fold : {self.model_current_path}")

            #Get the model
            model = self.get_model(fold=fold)

            #Flow the imagegenerator if required
            train_data_flow = self.flow_if_necessary(train_data)
            valid_data_flow = self.flow_if_necessary(validation_data)

            #Fit the model
            if not self.load:
                model.fit(
                    train_data_flow,
                    **self.fit_kwargs(valid_data_flow)
                )
            
            #Make a prediction
            self.predict(model, 
                validation_data,
                validation_data.targets,
                train_data)
            
            return model


    def predict(self, model, features, targets, generator):

        #Flow the imagegenerator if required
        features_flow = self.flow_if_necessary(features)

        y_pred = model.predict(features_flow)
        if self.model_neural:
            y_pred = np.argmax(y_pred, axis=1)

        y_true = targets

        score = balanced_accuracy_score(y_true, y_pred)

        print(y_pred)
        print(y_true)
            
        #Decode the targets
        y_pred = generator.decode(y_pred)
        y_true = generator.decode(y_true)

        #Build the classification report
        report = self.report_df(y_true, y_pred)

        #Build the crosstab
        self.crosstab(y_true, y_pred)

        #Push the score in summary
        self.save_summary(report)

        #Save summary
        self.save_model_summary(model)


        return score
        
    def crosstab(self, y_true, y_pred):
        #Calculate the crosstab, normalisé selon les colonnes
        crosstab = pd.crosstab(
            y_true, y_pred,
            rownames=['Realité'], colnames=['Prédiction'],
            normalize="index"
        )
        #Save it in a csv file
        #if self.save:
        path = self.model_current_path
        crosstab.to_csv(Path(path, f'crosstab_report.csv'), index= True)

    def report_df(self, y_true, y_pred):
        
        #Create clf report
        clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        #Create a df with the report
        clf_report = pd.DataFrame(clf_report).transpose()

        #Save the dataframe as csv file
        #if self.save:
        path = self.model_current_path
        clf_report.to_csv(Path(path, f'clf_report.csv'), index= True)
        
        #Return the clf report
        return clf_report


    def select_best_fold(self, scores, metric="recall", mode="max"):

        #Get the best fold
        best_fold = np.argmax(scores)

        #Get the best score
        best_score = np.max(scores)

        print(f"Best score of {best_score} for fold {best_fold} (other scores : {scores})")

        #Load the model with the best score
        return self.load_model(fold=best_fold)

    def save_model_summary(self, model):

        if self.model_neural:
            path = Path(self.model_current_path, "model_summary.txt")
            if path.exists():
                os.remove(path)

            #Save the model summary (layers) to a txt file
            with open(path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
                

    def save_summary(self, report, metrics=["precision", "recall", "f1-score"]):

        #For all metrics
        for metric in metrics:

            #Create a path to the summary file
            summary_path = Path(self.models_folder, self.type, f"summary_{metric}.csv")
            
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
            df[self.get_name()] = expected_values
            
            #Save it back to the summary file
            df.to_csv(summary_path)

    def decode_labels(self, labels, encoder=None):
        return encoder.inverse_transform(label)

