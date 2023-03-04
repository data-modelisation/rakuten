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

        return None

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
                call_dashboard(self.model_fold_path),
                call_checkpoint(self.model_fold_path),
                call_earlystopping(),
                call_reducelr(),
            ] if self.model_neural else []

    def get_path(self, fold=-1):
        return self.model_path if fold < 0 else self.model_fold_path
    
    def set_model_folder(self,):
        self.model_path = Path(self.models_folder, self.type , self.get_name())
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
            path = Path(self.model_fold_path)
        else :
            path = Path(self.model_path)

        if self.model_neural:
            model = tf.keras.models.load_model(path)#(Path(path, "saved_model.hdf5"))
        else:
            model = joblib.load(Path(path, "saved_model.pb"))
        
        print(f"model loaded from {path}")
        return model

    def save_model(self, model, in_fold=False):
        if self.save :
            path = self.model_path if not in_fold else self.model_fold_path
            #joblib.dump(model, Path(path, 'model.joblib'))
            
            if self.model_neural:
                model.save(Path(path))
            else:
                joblib.dump(model, Path(path, "saved_model.pb"))
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


    def unpack_iterator(self, iterator):
        *unpacked, = iterator
        x = []
        y = []
        for array in unpacked:
            x.append(array[0])
            y.append(array[1])
        return np.concatenate(x), np.concatenate(y)

    def set_folder(self, fold=-1):
        #If there is no model path, we create it
        if not self.model_path:
            self.set_model_folder()

        #If there is no fold path, we create it
        if not fold < 0:
            self.set_model_fold_folder(fold)


    def flow_if_necessary(self, input):
        
        return input.flow() if isinstance(input, ImageGenerator) else input

    def kfit(self, model=None, train_data=None, validation_data=None, fold=-1, class_weight=None, force=False):
        

        self.class_weight = class_weight if class_weight is not None else train_data.class_weight 

        #if isinstance(train_data, keras.preprocessing.image.DataFrameIterator):
            

        #If it's a fusion model, we build a special sequence with texts and images
        if self.type == "fusion" and not isinstance(train_data, FusionGenerator):
            train_data = FusionGenerator(train_data)
            validation_data = FusionGenerator(validation_data)

            print("data transformed to sequences for fusion")
            fold = 1



        #Get the model
        

        #If fold is negtiv, we perform the cross validation process
        if fold < 0:

            #Save the scores
            scores = []

            #Run the cross validation process (pas besoin de random_state car on a pas suffle)
            for idx_fold, (train_index, valid_index) in enumerate(KFold(self.num_folds, ).split(train_data)):
                
                #Creation du dossier
                self.set_folder(fold=idx_fold)

                print("indexes", train_index)
                #Split
                train_gen, valid_gen = train_data.split(split_indexes=[train_index, valid_index], is_batch=True)
                
                train_gen_flow = self.flow_if_necessary(train_gen)
                valid_gen_flow = self.flow_if_necessary(valid_gen)
                
                model = self.get_model(fold=idx_fold)

                if not self.load:
                    model.fit(
                        train_gen_flow,
                        **self.fit_kwargs(valid_gen_flow)
                    )

                y_pred = model.predict(valid_gen_flow)

                y_pred = np.argmax(y_pred, axis=1)
                y_true = valid_gen.targets

                y_pred = train_data.decode(y_pred)
                y_true = train_data.decode(y_true)

                #Get the classification report
                report = self.report_df(y_true, y_pred, fold=idx_fold)

                #Save the crosstab
                self.crosstab(y_true, y_pred, fold=idx_fold)

                score = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                scores.append(score)
                self.save_model(model, in_fold=True)

                #Save it in the summary if asked
                if self.summary:
                    self.save_summary(report)

            self.model = self.select_best_fold(scores)
            self.save_model(self.model, in_fold=False)
            train_gen_flow = self.flow_if_necessary(train_data)
            valid_gen_flow = self.flow_if_necessary(validation_data)    

            y_pred = self.model.predict(valid_gen_flow)

            y_pred = np.argmax(y_pred, axis=1)
            y_true = validation_data.targets

            y_pred = train_data.decode(y_pred)
            y_true = train_data.decode(y_true)

            #Get the classification report
            report = self.report_df(y_true, y_pred)

            #Save the crosstab
            self.crosstab(y_true, y_pred)

            #Save it in the summary if asked
            if self.summary:
                self.save_summary(report)

            #     #Call the training
            #     model = self.kfit(
            #         train_data=train_gen,
            #         validation_data=valid_gen,
            #         fold=idx_fold
            #     )
                            
            #     #Make prediction on the validation dataset
            #     if self.model_neural:
            #         y_pred = self.predict(generator=valid_gen, targets=valid_gen.targets, model=model, fold=idx_fold)
            #     else:
            #         valid_gen_unpacked, _ = self.unpack_iterator(valid_gen)
            #         y_pred = self.predict(features=valid_gen_unpacked, generator=valid_gen, targets=valid_gen.targets, model=model, fold=idx_fold)
                    
            #     y_true = valid_gen.decode(valid_gen.targets)
            #     #Calculation of the recall score
            #     import pdb; pdb.set_trace()
            #     score = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            #     scores.append(score) 

            
            # #Select the best model an push it as attribute
            # self.model = self.select_best_fold(scores)
            
            # #Save the best model in the main folder
            # self.save_model(self.model, in_fold=False)

            # #Make prediction on the test dataset with the best model
            # print("prediction with best model")
            # if self.model_neural:
            #     y_pred = self.predict(generator=validation_data, targets=validation_data.targets, model=self.model)
            # else:
            #     validation_data_unpacked, _ = self.unpack_iterator(validation_data)
            #     y_pred = self.predict(features=validation_data_unpacked, generator=validation_data, targets=validation_data.targets, model=model)
              

        #If fold is positiv, we make the fit
        else:
            print(f"starting fit process for fold {fold} : {self.type}")

            if not self.load:

                #Fitting the model to the features for neural
                if self.model_neural:
                    train_data_flow = self.flow_if_necessary(train_data)
                    validation_data_flow = self.flow_if_necessary(validation_data)
                    
                    model.fit(train_data_flow, **self.fit_kwargs(validation_data_flow))
                #Fitting the model to the features for non neural
                else:
                    train_data, validation_data = self.unpack_iterator(train_data)
                    model.fit(train_data, validation_data)
                
                print(f"end of fit in fold {fold}")
            
                # Save model in the kfold folder
                self.save_model(model, in_fold=True)

                # Save the layers in the model summary file
                self.save_model_summary(model)

            if force:
                y_pred = self.predict(generator=validation_data, targets=validation_data.targets, model=model)

            self.model = model
            #Return the model
            return model

    def predict(self, features=None, generator=None, targets=None, model=None, fold=-1):

        print("prediction")
        #Get the model
        model = model if model is not None else self.model

        #Predict the targets
        if features is not None:
            y_pred = model.predict(features)
        else:
            generator_flowed = self.flow_if_necessary(generator)
            y_pred = model.predict(generator_flowed)

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

            #Save the crosstab
            self.crosstab(y_true, y_pred, fold=fold)

            #Save it in the summary if asked
            if self.summary:
                self.save_summary(report)
        
        return y_pred
        
    def crosstab(self, y_true, y_pred, fold=-1):
        #Calculate the crosstab, normalisé selon les colonnes
        crosstab = pd.crosstab(
            y_true, y_pred,
            rownames=['Realité'], colnames=['Prédiction'],
            normalize="index"
        )
        #Save it in a csv file
        #if self.save:
        path = self.get_path(fold)
        crosstab.to_csv(Path(path, f'crosstab_report.csv'), index= True)

    def report_df(self, y_true, y_pred, fold=-1):
        
        #Create clf report
        clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        #Create a df with the report
        clf_report = pd.DataFrame(clf_report).transpose()

        #Save the dataframe as csv file
        #if self.save:
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

    def save_model_summary(self, model):

        if self.model_neural:
            path = Path(self.model_path, "model_summary.txt")
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

