import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, recall_score,classification_report
from imblearn.metrics import classification_report_imbalanced

from src.models.models_utils import call_memory, call_tqdm, call_dashboard, call_checkpoint, call_earlystopping, call_reducelr
from src.generators.generator_fusion import FusionGenerator

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
        folder="src/models/",
        ):

        self.preprocessor = None
        self.preprocessor_fitted = False
        self.model_fitted = False
        self.model = None
        
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

        return None

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
                call_dashboard(self.model_fold_path),
                call_checkpoint(self.model_fold_path),
                call_earlystopping(),
                call_reducelr(),
            ] if self.model_neural else []

    def get_path(self, fold=-1):
        return self.model_path if fold < 0 else self.model_fold_path
    
    def set_model_folder(self,):
        self.model_path = Path(self.models_folder, self.type , self.name)
        self.model_path.mkdir(parents=True, exist_ok=True)
        print(f"{self.model_path} created")
  
    def set_model_fold_folder(self, fold=0):

        if self.model_path is None:
            self.set_model_folder()

        self.model_fold_path = Path(self.model_path, "folds", f"fold_{fold:02d}")
        self.model_fold_path.mkdir(parents=True, exist_ok=True)
        print(f"{self.model_fold_path} created")

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
            
    def load_model(self, fold=-1):

        if self.model_path is None:
            self.set_model_folder()

        if not fold < 0:
            
            self.set_model_fold_folder(fold=fold)
            path = Path(self.model_fold_path, 'model.joblib')
        else :
            path = Path(self.model_path, 'model.joblib')

        print(f"model loaded from {path}")
        return joblib.load(path)
        

    def save_model(self, model, in_fold=False):
        if self.save :
            path = self.model_path if not in_fold else self.model_fold_path
            joblib.dump(model, Path(path, 'model.joblib'))
            print(f"model saved in {path}")

    def fit_preprocessor(self, values):

        if not self.preprocessor:
            self.preprocessor = self.get_preprocessor()

        values_transformed = self.preprocessor.fit_transform(values) if self.preprocessor else values

        self.preprocessor_fitted = True
        if self.preprocessor:
            self.save_preprocess()

        return values_transformed

    def get_model(self, fold=-1):
        if self.load:                
            return self.load_model(fold=fold)
        else:
            return self.init_model()

    def kfit(self, model=None, train_data=None, validation_data=None, fold=-1):
        
        #If it's a fusion model, we build a special sequence with texts and images
        if self.type == "fusion" and not isinstance(train_data, FusionGenerator):
            train_data = FusionGenerator(train_data)
            validation_data = FusionGenerator(validation_data)
            print("data transformed to sequences for fusion")

        #If there is no model path, we create it
        if not self.model_path:
            self.set_model_folder()

        #If there is no fold path, we create it
        if not fold < 0:
            self.set_model_fold_folder(fold)

        #Get the model
        model = self.get_model(fold=fold)

        self.class_weight = train_data.class_weight

        #If fold is negtiv, we perform the cross validation process
        if fold < 0:

            #Save the scores
            scores = []

            #Run the cross validation process
            for idx_fold, (train_index, valid_index) in enumerate(KFold(self.num_folds).split(train_data)):
                
                #Split
                train_gen, valid_gen = train_data.split(split_indexes=[train_index, valid_index], is_batch=True)

                #Call the training
                model = self.kfit(
                    train_data=train_gen,
                    validation_data=valid_gen,
                    fold=idx_fold
                )

                #Make prediction on the validation dataset
                y_pred = self.predict(valid_gen, targets=valid_gen.targets, model=model, fold=idx_fold)
                y_true = valid_gen.decode(valid_gen.targets)

                #Calculation of the recall score
                score = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                scores.append(score)      
            
            #Select the best model an push it as attribute
            self.model = self.select_best_fold(scores)
            
            #Save the best model as
            if not self.load:
                self.save_model(self.model, in_fold=False)

            #Make prediction on the test dataset with the best model
            y_pred = self.predict(validation_data, targets=validation_data.targets, model=self.model)

        #If fold is positiv, we make the fit
        else:
            print(f"starting fit process for fold {fold} : {self.type}")

            if not self.load:
                #Fitting the model to the features
                model.fit(
                    train_data,
                    **self.fit_kwargs(validation_data)
                )

                print(f"end of fit in fold {fold}")
            
                self.save_model(model, in_fold=True)

            #Return the model
            return model

    def predict(self, generator=None, targets=None, model=None, fold=-1):

        #Get the model
        model = model if model is not None else self.model

        #Predict the targets
        y_pred = model.predict(generator)
        
        #Get the class if neural
        if self.model_neural:
            y_pred = np.argmax(y_pred, axis=1)
        
        #Decode
        y_pred = generator.decode(y_pred)
        
        #If there is a target, we can make a report
        if targets is not None:

            #Decode targets
            y_true = generator.decode(targets)

            #Get the classification report
            report = self.report_df(y_true, y_pred, fold=fold)

            #Save it in the summary if asked
            if self.summary:
                self.save_summary(report)
        
        return y_pred
        
        
    def report_df(self, y_true, y_pred, fold=-1):
        
        #Create clf report
        clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        #Create a df with the report
        clf_report = pd.DataFrame(clf_report).transpose()

        #Save the dataframe as csv file
        if self.save:
            path = self.get_path(fold)
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
            df[self.name] = expected_values
            
            #Save it back to the summary file
            df.to_csv(summary_path)

    def decode_labels(self, labels, encoder=None):
        return encoder.inverse_transform(label)

