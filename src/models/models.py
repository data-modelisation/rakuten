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
#from src.generators.generator_fusion import FusionGenerator
#from src.generators.generator_image import ImageGenerator
from src.models.models_utils import METRICS


COLUMNS = [10,40,50,60,1140,1160,1180,1280,1281,1300,1301,1302,1320,1560,1920,1940,2060,2220,2280,2403,2462,2522,2582,2583,2585,2705,2905,]

def load_image(x):

    from main import TARGET_SHAPE

    im = tf.io.read_file(x[0])
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.resize(im, size=TARGET_SHAPE[:2])

    return im
    
def load_image_test(x):

    from main import TARGET_SHAPE

    im = tf.io.read_file(str(x))
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.resize(im, size=TARGET_SHAPE[:2])

    return im

class MyModel():

    def __init__(self,
        num_folds=3,
        load=False,
        suffix="",
        models_folder="src/models",
        batch_size=32,
        epochs=30,
        class_weight=None):

        # Attributs
        self.batch_size = batch_size
        self.epochs = epochs
        self.load = load
        self.num_folds = 3
        self.model = None
        self.model_path = None
        self.models_folder = models_folder
        self.suffix = suffix
        self.preprocessor = None
        self.class_weight = class_weight
        self.preprocessor_fitted = False
        

    @property
    def compilation_kwargs(self):
        """Configuration parameters for neural networks"""

        return {
            "loss" : 'sparse_categorical_crossentropy', 
            "optimizer" : 'adam', 
            "metrics" :METRICS
        }

    @property
    def callbacks(self):
        """Renvoie les callbacks"""

        return [
                call_memory(),
                call_tqdm(),
                call_dashboard(self.model_current_path),
                call_checkpoint(self.model_current_path),
                call_earlystopping(),
                call_reducelr(),
            ] if self.model_neural else []

    def fit_kwargs(self, validation_data=None):
        """Dictionnaire des modèles"""

        return {
                "batch_size":self.batch_size,
                "epochs" : self.epochs,
                "validation_data" : validation_data,
                "verbose":0,
                "class_weight" : self.class_weight, #TODO
                "callbacks" : self.callbacks
            } if self.model_neural else {}

    def set_current_folder(self, fold=None):
        """Chemin vers le dossier du modèle actuel"""

        #Si le chemin model_path n'existe pas, on le créé
        if self.model_path is None:
            self.set_model_path()

        #Si on est pas dans un fold, on renvoie le dossier model_path
        if fold is None or fold < -1:
            self.model_current_path = self.model_path

        #Sinon on choisit le dossier fold
        else:
            self.model_current_path = Path(self.model_path, "folds", f"fold_{fold:02d}")

        #On créé le dossier s'il n'existe pas
        self.model_current_path.mkdir(exist_ok=True, parents=True)

    def save_preprocessor(self, preprocessor):
        """Enregistre le preprocesseur du modèle"""
        
        if self.model_path is None:
            self.set_model_path()

        joblib.dump(preprocessor, Path(self.model_path, 'preprocess.joblib'))
        print(f"preprocessor saved in {self.model_path}")

    def preprocess(self, data_generator, preprocessor=None, type_=None):
        """Preprocess the datagenerator"""

        if type_ is None:
            type_ = self.model_type
        
        print(preprocessor)
        if preprocessor is None:
            preprocessor = self.get_preprocessor()
            preprocessor_fitted = self.preprocessor_fitted
        else : 
            preprocessor_fitted  = True


        if type_=="text" or type_=="image" :
            print("data not transformed yet")
            
            if preprocessor is not None:
                print(f"preprocess for {type_}")
                if not preprocessor_fitted:
                    print("preprocessor not fitted yet")
        
                    preprocessor.fit(data_generator.data.values)
                    print("processor fitted")
                    self.save_preprocessor(preprocessor)
                    print("preprocessor saved")

            print("transforming the data")
            return data_generator.preprocess(preprocessor, type_=type_)

        else:
            print("in rpeprocessloop")
            data_preprocessed = []
            for idx, model in enumerate(self.models):
                print(model.model_type)
                self.preprocess(
                    data_generator,
                    preprocessor=model.preprocessor,
                    type_=model.model_type
                )
                
                print(f"model {idx}")
                if idx < len(self.models)-1:
                    print("not last model")
                    data=data_generator.data_transformed[:, :-1]
                else:
                    print("last model")
                    data=data_generator.data_transformed
                data_preprocessed.append(data)
            data_generator.data_transformed = np.concatenate(data_preprocessed, axis=1)
            print(data_generator.data_transformed.shape)
            return data_generator

            
            

        

    def fit(self, train_data, validation_data=None, crossval=True, fold=None):
        
        #If a crossvalidation is to be done
        if crossval:
            scores = []

            for kfold, (train_index, valid_index) in enumerate(KFold(self.num_folds).split(train_data.data)):
                
                #Split into train and validation
                kdata_train, kdata_validation = train_data.split(indexes=[train_index, valid_index])               

                #Call the function itself to fit the model for the current fold
                _, kscore = self.fit(
                    kdata_train,
                    validation_data=kdata_validation,
                    crossval=False,
                    fold=kfold
                )

                #Push its score into a list
                scores.append(kscore)
            
            #Copy the best to the model root folder
            self.model = self.select_best_fold(scores)

            self.set_current_folder(fold=None)
            self.save_model(self.model)
            self.save_model_summary(self.model)

            # Preprocess the data
            train_data = self.preprocess(train_data)
            validation_data = self.preprocess(validation_data)


            train_dataset, validation_dataset = self.get_datasets(train_data, validation_data)
            if self.model_neural:
                self.make_prediction(validation_dataset, validation_data)
            else:
                self.make_prediction(validation_dataset[0], validation_data)

        #Otherwise, the model is fitted
        else:
            print(f"start fit on fold {fold} for {self.__class__}")
            self.set_current_folder(fold=fold)
            
            self.model = self.get_model(fold=fold)
            
            #Compilation of the model
            self.compile()

            # Preprocess the data
            train_data = self.preprocess(train_data)
            validation_data = self.preprocess(validation_data)

            #Transform the data in datasets
            train_dataset, validation_dataset = self.get_datasets(train_data, validation_data)
            
            #For neural networks
            if self.model_neural:
                if not self.load:
                    self.model.fit(
                        train_dataset,
                        **self.fit_kwargs(validation_dataset)
                    )
                else:
                    print("model already fitted")
                score = self.make_prediction(validation_dataset, validation_data)
            
            #For sklearn models
            else:
                if not self.load:
                    self.model.fit(train_dataset[0], train_dataset[1])
                else:
                    print("model already fitted")
                score = self.make_prediction(validation_dataset[0], validation_data)

            self.save_model(self.model)
            self.save_model_summary(self.model)

            return self.model, score

    def get_datasets(self, train_data, validation_data):
        print(f'conversion to datasets for {self.model_type} : neural {self.model_neural}') 
        #Neural
        if self.model_neural:

            if self.model_type == "text":
                X_train_text, y_train_text = train_data.data_transformed[:,:-1], train_data.data_transformed[:,-1]
                X_valid_text, y_valid_text = validation_data.data_transformed[:,:-1], validation_data.data_transformed[:,-1]

                y_train_text = np.asarray(y_train_text).astype('float32')
                y_valid_text = np.asarray(y_valid_text).astype('float32')

                dataset_train_text = tf.data.Dataset.from_tensor_slices((X_train_text, y_train_text))
                dataset_validation_text = tf.data.Dataset.from_tensor_slices((X_valid_text, y_valid_text))
                
                #Batch the dataset
                dataset_train_text = dataset_train_text.batch(self.batch_size)
                dataset_validation_text = dataset_validation_text.batch(self.batch_size)

                if self.model_type == "text":
                    return dataset_train_text, dataset_validation_text

            if self.model_type == "image":
                X_train_image, y_train_image = train_data.data_transformed[:,:1], train_data.data_transformed[:,-1]
                X_valid_image, y_valid_image = validation_data.data_transformed[:,:1], validation_data.data_transformed[:,-1]

                y_train_image = np.asarray(y_train_image).astype('float32')
                y_valid_image = np.asarray(y_valid_image).astype('float32')

                dataset_train_image = tf.data.Dataset.from_tensor_slices((X_train_image, y_train_image))
                dataset_validation_image = tf.data.Dataset.from_tensor_slices((X_valid_image, y_valid_image))

                dataset_train_image = dataset_train_image.map(lambda x,y: [load_image(x), y])
                dataset_validation_image = dataset_validation_image.map(lambda x,y: [load_image(x), y])

                #Batch the dataset
                dataset_train_image = dataset_train_image.batch(self.batch_size)
                dataset_validation_image = dataset_validation_image.batch(self.batch_size)

                if self.model_type == "image":
                    return dataset_train_image, dataset_validation_image

            if self.model_type == "fusion":
                
                INPUT_SHAPE_IMAGE = [50,50,3]
                INPUT_SHAPE_TEXT = [1, 200]
                def gen_text():
                    for i in range(1000):
                        im = np.random.rand(INPUT_SHAPE_IMAGE)
                        text = np.random.rand(INPUT_SHAPE_TEXT)
                        y = np.random.randint(0, 27, size=1)
                        yield {"text_input_input": text}, y
                def gen_image():
                    for i in range(1000):
                        im = np.random.rand(INPUT_SHAPE_IMAGE)
                        text = np.random.rand(INPUT_SHAPE_TEXT)
                        y = np.random.randint(0, 27, size=1)
                        yield {"image_input": im}, y

                dataset_train_text = tf.data.Dataset.from_generator(gen_text, 
                                         (tf.float32, tf.float32, ),
                                         output_shapes=( tf.TensorShape(INPUT_SHAPE_TEXT), tf.TensorShape(None))
                                        )
                dataset_train_image = tf.data.Dataset.from_generator(gen_image, 
                                         (tf.float32, tf.float32, ),
                                         output_shapes=( tf.TensorShape(INPUT_SHAPE_IMAGE), tf.TensorShape(None))
                                        )
                
                # X_train_text, y_train_text = train_data.data_transformed[:,1:-1], train_data.data_transformed[:,-1]
                # X_valid_text, y_valid_text = validation_data.data_transformed[:,1:-1], validation_data.data_transformed[:,-1]

                # X_train_image, y_train_image = train_data.data_transformed[:,:1], train_data.data_transformed[:,-1]
                # X_valid_image, y_valid_image = validation_data.data_transformed[:,:1], validation_data.data_transformed[:,-1]

                # y_train = np.asarray(y_train_text).astype(np.float32)
                # y_valid = np.asarray(y_valid_text).astype(np.float32)
                # X_train_text = np.asarray(X_train_text).astype(np.float32)
                # X_valid_text = np.asarray(X_valid_text).astype(np.float32)

                # dataset_train_image = tf.data.Dataset.from_tensor_slices((X_train_image))
                # dataset_validation_image = tf.data.Dataset.from_tensor_slices((X_valid_image))
                
                # dataset_train_text = tf.data.Dataset.from_tensor_slices(X_train_text)
                # dataset_validation_text = tf.data.Dataset.from_tensor_slices(X_valid_text)

                # dataset_train_image = dataset_train_image.map(lambda x: load_image_test(x))
                # dataset_validation_image = dataset_validation_image.map(lambda x: load_image_test(x))


                # dataset_train_fusion = tf.data.Dataset.zip((dataset_train_image, dataset_train_text))
                # dataset_validation_fusion = tf.data.Dataset.zip((dataset_validation_image, dataset_validation_text))

                # def generator_train():
                #     for s1, s2, l in zip(X_train_image, X_train_text, y_train):
                #         print(s1, s2, l)
                #         yield {"image_input": np.asarray(load_image(s1)), "text_input_input": np.asarray(s2)}, l.set_shape([1])
                # def generator_validation():
                #     for s1, s2, l in zip(X_valid_image, X_valid_text, y_valid):
                #         print(s1, s2, l)
                #         yield {"image_input": np.asarray(load_image(s1)), "text_input_input": np.asarray(s2)}, l.set_shape([1])
                
                #import pdb; pdb.set_trace()
                # dataset_train_fusion = tf.data.Dataset.from_generator(generator_train, output_types=({"image_input": np.float32, "text_input_input": np.float32}, np.float32))
                # dataset_validation_fusion = tf.data.Dataset.from_generator(generator_validation, output_types=({"image_input": np.float32, "text_input_input": np.float32}, np.float32))

                #dataset_train_fusion = dataset_train_fusion.batch(7).padded_batch(7, padded_shapes=([None]))
                #dataset_validation_fusion = dataset_validation_fusion.batch(7).padded_batch(7, padded_shapes=([None]))
  

                # dataset_train_fusion = dataset_train_fusion.batch(self.batch_size)
                # dataset_validation_fusion = dataset_validation_fusion.batch(self.batch_size)

                return (dataset_train_image, dataset_train_text), (None, None)


        #Not neural
        else:
            #Text
            if (self.model_type == "text"):
                dataset_train = train_data.data_transformed[:,:-1], train_data.data_transformed[:,-1]
                dataset_validation = validation_data.data_transformed[:,:-1], validation_data.data_transformed[:,-1]
            #Image
            elif (self.model_type == "image"):
                dataset_train = train_data.data_transformed[:,1].reshape(-1, 1), train_data.data_transformed[:,-1]
                dataset_validation = validation_data.data_transformed[:,1].reshape(-1, 1), validation_data.data_transformed[:,-1]

            print(f'datasets created for {self.model_type}')     
            return dataset_train, dataset_validation

    def make_prediction(self, prediction_features, generator, decode=True):
        
        #Make a prediction
        y_pred = self.model.predict(prediction_features)

        #If the model is neural, get the classes
        if self.model_neural:
            y_pred = np.argmax(y_pred, axis=1)

        # Get the true values
        y_true = generator.data_transformed[:, -1]

        #If we want to go back to the original categories
        if decode:
            y_true = generator.decode(y_true)
            y_pred = generator.decode(y_pred)

        #Small conversion to int
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        #Make a classification report
        report = self.save_classification_report(y_true, y_pred)

        #Save the report to the summary
        self.save_in_summary(report)

        #Save the crosstab
        self.save_crosstab(y_true, y_pred)

        #Return the score
        return balanced_accuracy_score(y_true, y_pred)

    def set_model_path(self,):
        
        self.model_path = Path(self.models_folder, self.model_type, self.get_name())
        self.model_path.mkdir(exist_ok=True, parents=True)

    def get_name(self):
        """Return the name of the model and its suffix"""
        return self.model_name + self.suffix

    def select_best_fold(self, scores, metric="recall", mode="max"):
        """Selection of the best model after crossvalidation"""

        #Get the best fold
        best_fold = np.argmax(scores)

        #Get the best score
        best_score = np.max(scores)

        print(f"Best score of {best_score} for fold {best_fold} (other scores : {scores})")

        #Load the model with the best score
        return self.load_model(fold=best_fold)

    def load_preprocessor(self):
        if self.model_path is None:
            self.set_model_path()
        try:
            path = Path(self.model_path, 'preprocess.joblib')
            preprocessor = joblib.load(path)
            self.preprocessor_fitted = True
            print(f"processor loaded from {path}")
        except:
            print(f"unable to find a preprocessor {path}")
            self.preprocessor_fitted = False
            preprocessor = self.init_preprocessor()
            
        return preprocessor

    def load_model(self, fold=None):
        
        self.set_current_folder(fold=fold)

        print(self.model_current_path)
        if self.model_neural:
            try:
                model = tf.keras.models.load_model(self.model_current_path)#(Path(path, "saved_model.hdf5"))
            except:
                try:
                    model = tf.keras.models.load_model(Path(self.model_current_path, "saved_model.hdf5"))#(Path(path, "saved_model.hdf5"))
                except:
                    model = joblib.load(Path(self.model_current_path, "saved_model.pb"))
                        
        else:
            model = joblib.load(Path(self.model_current_path, "saved_model.pb"))
        
        print(f"model loaded from {self.model_current_path}")
        return model

    def get_model(self, fold=None):
        self.model = self.init_model()     
        return self.load_model(fold=fold) if self.load else self.model

    def save_model(self, model, in_fold=False):
        if self.model_neural:
            model.save(self.model_current_path)
        else:
            joblib.dump(model, Path(self.model_current_path, "saved_model.pb"))
        print(f"model saved in {self.model_current_path}")

    def get_preprocessor(self):

        try:               
            return self.load_preprocessor()
        except Exception as exce:
            print(exce)
            print("unable to load the preprocessor, default one init")
            self.preprocessor_fitted = False
            return self.init_preprocessor()


    @property
    def model_name(self,):
        """Extract the model name from it class name"""
        return type(self).__name__.lower().replace("model", "")

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
    def model_neural(self):
        """Check if the model is neural"""
        #Neural if there it's a sequential model or a functional one
        is_seq =  isinstance(self.model, keras.engine.sequential.Sequential)
        is_fon =  isinstance(self.model, keras.engine.functional.Functional)
        return is_seq or is_fon

    def compile(self,):
        """"Call the comppilation of the model"""

        #If the model is neural, compile it with its parameters (loss, metrics, etc..)
        if self.model_neural:
            self.model.compile(**self.compilation_kwargs)

    def save_model_summary(self, model):
        """Save the description of the layers of the model into a file"""

        #If it's a neural network
        if self.model_neural:
            path = Path(self.model_current_path, "model_summary.txt")

            #Save the model summary (layers) to a txt file
            with open(path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
                
    def save_crosstab(self, y_true, y_pred):
        """"Calculate the crosstab between targets and predictions"""

        #Calculate the crosstab, normalisé selon les colonnes
        crosstab = pd.crosstab(
            y_true, y_pred,
            rownames=['Realité'], colnames=['Prédiction'],
            normalize="index"
        )
        #Save it in a csv file
        path = self.model_current_path
        crosstab.to_csv(Path(path, f'crosstab_report.csv'), index= True)

    def save_classification_report(self, y_true, y_pred):
        """Create and save a classification report"""

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

    def save_in_summary(self, report, metrics=["precision", "recall", "f1-score"]):
        """Push the classification_report values to the summary"""
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
            df[self.get_name()] = expected_values
            
            #Save it back to the summary file
            df.to_csv(summary_path)

# class Model():

#     def __init__(self, 
#         generator = None,
#         generator_test = None,
#         num_folds=3, 
#         epochs=2,
#         batch_size=10,
#         load=True,   
#         save=True,
#         report=True,
#         summary=True,
#         predict=False,
#         folder="src/models/",
#         name=None,
#         suffix="",
#         random_state=42,
#         ):

#         self.preprocessor = None
#         self.preprocessor_fitted = False
#         self.model_fitted = False
#         self.model = None
#         self.suffix=suffix
#         self.random_state=random_state
        
#         self.generator = generator
#         self.generator_test = generator_test
        
#         self.num_folds = num_folds
#         self.models_folder = folder
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.summary = summary
#         self.report=report
#         self.load = load
#         self.save = save
#         self.model_path = None
#         self.model_current_path = None

#         return None


#     def select_best_fold(self, scores, metric="recall", mode="max"):

#         #Get the best fold
#         best_fold = np.argmax(scores)

#         #Get the best score
#         best_score = np.max(scores)

#         print(f"Best score of {best_score} for fold {best_fold} (other scores : {scores})")

#         #Load the model with the best score
#         return self.load_model(fold=best_fold)

#     def set_model_path(self,):
#         self.model_path = Path(self.models_folder, self.type, self.get_name())

#     def get_name(self):
#         return self.name + self.suffix

#     def fit_kwargs(self, validation_data=None):

#         return {
#                 "batch_size":self.batch_size,
#                 "epochs" : self.epochs,
#                 "validation_data" : validation_data,
#                 "verbose":0,
#                 "class_weight" : self.class_weight,
#                 "callbacks" : self.callbacks
#             } if self.model_neural else {}
            
#     @property
#     def has_preprocessor(self):
#         return self.get_preprocessor() is not None

#     @property
#     def callbacks(self):
#         return [
#                 call_memory(),
#                 call_tqdm(),
#                 call_dashboard(self.model_current_path),
#                 call_checkpoint(self.model_current_path),
#                 call_earlystopping(),
#                 call_reducelr(),
#             ] if self.model_neural else []

    

#     def load_preprocess(self):
#         if self.model_path is None:
#             self.set_model_folder()

#         if self.has_preprocessor:
#             self.preprocessor = joblib.load(Path(self.model_path, 'preprocess.joblib'))
            
#         self.preprocessor_fitted = True


#     def save_preprocess(self):
#         if self.preprocessor_fitted and self.save:
#             path = Path(self.model_path, 'preprocess.joblib')
#             joblib.dump(self.preprocessor, path)
#             print(f"preprocess savec in {path}")
            
#     def load_model(self, fold=None):
        
#         self.set_current_folder(fold=fold)

#         if self.model_neural:
#             model = tf.keras.models.load_model(self.model_current_path)#(Path(path, "saved_model.hdf5"))
#         else:
#             model = joblib.load(Path(self.model_current_path, "saved_model.pb"))
        
#         print(f"model loaded from {self.model_current_path}")
#         return model

#     def save_model(self, model, in_fold=False):
#         if self.model_neural:
#             model.save(self.model_current_path)
#         else:
#             joblib.dump(model, Path(self.model_current_path, "saved_model.pb"))
#         print(f"model saved in {self.model_current_path}")

#     def fit_preprocessor(self, values):

#         if not self.preprocessor:
#             self.preprocessor = self.get_preprocessor()

#         values_transformed = self.preprocessor.fit_transform(values) if self.preprocessor else values

#         self.preprocessor_fitted = True
#         if self.preprocessor:
#             self.save_preprocess()

#         return values_transformed

#     def get_model(self, fold=None):
#         if self.load:                
#             return self.load_model(fold=fold)
#         else:
#             return self.init_model()


#     def unpack_iterator(self, iterator):
#         *unpacked, = iterator
#         x = []
#         y = []
#         for array in unpacked:
#             x.append(array[0])
#             y.append(array[1])
#         return np.concatenate(x), np.concatenate(y)

#     def set_current_folder(self, fold=None):
        
#         if self.model_path is None:
#             self.set_model_path()


#         if fold is None or fold < -1:
#             self.model_current_path = self.model_path
#         else:
#             self.model_current_path = Path(self.model_path, "folds", f"fold_{fold:02d}")

#         self.model_current_path.mkdir(exist_ok=True, parents=True)
 

#     def flow_if_necessary(self, input, type="validation"):
        
#         return input.flow(type_=type) if isinstance(input, ImageGenerator) else input


#     def fit(self, model=None, train_data=None, validation_data=None, fold=None, crossval=True, class_weight=None):
        
#         #Set the weights
#         self.class_weight = class_weight if class_weight is not None else train_data.class_weight 

#         #If it's a fusion model, we build a special sequence with texts and images
#         if self.type == "fusion" and not isinstance(train_data, FusionGenerator):
#             train_data = FusionGenerator(train_data)
#             validation_data = FusionGenerator(validation_data)

#             print("data transformed to sequences for fusion")
#             fold = 1
      
#         if crossval:
#             scores = []

#             for kfold, (train_index, valid_index) in enumerate(KFold(self.num_folds).split(train_data)):

#                 kfold_train_data = train_data 
#                 kfold_valid_data = validation_data 
                
#                 # Entrainement sur les donne entrainement
#                 model = self.fit( 
#                     train_data=kfold_train_data, 
#                     validation_data=kfold_valid_data, 
#                     fold=kfold,
#                     crossval=False, 
#                     class_weight=class_weight)

#                 #Make a prediciton
#                 kscore = self.predict(model, 
#                     validation_data,
#                     validation_data.targets,
#                     train_data)
                
#                 scores.append(kscore)
            
#             #Copy the best to the model root folder
#             model = self.select_best_fold(
#                 scores
#             )

#             self.set_current_folder(fold=None)
#             self.save_model(model)

#             #Make a prediction
#             self.predict(model, 
#                 validation_data,
#                 validation_data.targets,
#                 train_data)
        
#         else:

#             #Set the folder
#             self.set_current_folder(fold=fold)
#             print(f"Current fold : {self.model_current_path}")

#             #Get the model
#             model = self.get_model(fold=fold)

#             #Flow the imagegenerator if required
#             train_data_flow = self.flow_if_necessary(train_data)
#             valid_data_flow = self.flow_if_necessary(validation_data)

#             #Fit the model
#             if not self.load:
#                 model.fit(
#                     train_data_flow,
#                     **self.fit_kwargs(valid_data_flow)
#                 )
            
#             #Make a prediction
#             self.predict(model, 
#                 validation_data,
#                 validation_data.targets,
#                 train_data)
            
#             return model


#     def predict(self, model, features, targets, generator):

#         #Flow the imagegenerator if required
#         features_flow = self.flow_if_necessary(features)

#         y_pred = model.predict(features_flow)
#         if self.model_neural:
#             y_pred = np.argmax(y_pred, axis=1)

#         y_true = targets

#         score = balanced_accuracy_score(y_true, y_pred)

#         print(y_pred)
#         print(y_true)
            
#         #Decode the targets
#         y_pred = generator.decode(y_pred)
#         y_true = generator.decode(y_true)

#         #Build the classification report
#         report = self.report_df(y_true, y_pred)

#         #Build the crosstab
#         self.crosstab(y_true, y_pred)

#         #Push the score in summary
#         self.save_summary(report)

#         #Save summary
#         self.save_model_summary(model)


#         return score
        
#     def crosstab(self, y_true, y_pred):
#         #Calculate the crosstab, normalisé selon les colonnes
#         crosstab = pd.crosstab(
#             y_true, y_pred,
#             rownames=['Realité'], colnames=['Prédiction'],
#             normalize="index"
#         )
#         #Save it in a csv file
#         #if self.save:
#         path = self.model_current_path
#         crosstab.to_csv(Path(path, f'crosstab_report.csv'), index= True)

#     def report_df(self, y_true, y_pred):
        
#         #Create clf report
#         clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
#         #Create a df with the report
#         clf_report = pd.DataFrame(clf_report).transpose()

#         #Save the dataframe as csv file
#         #if self.save:
#         path = self.model_current_path
#         clf_report.to_csv(Path(path, f'clf_report.csv'), index= True)
        
#         #Return the clf report
#         return clf_report


#     def select_best_fold(self, scores, metric="recall", mode="max"):

#         #Get the best fold
#         best_fold = np.argmax(scores)

#         #Get the best score
#         best_score = np.max(scores)

#         print(f"Best score of {best_score} for fold {best_fold} (other scores : {scores})")

#         #Load the model with the best score
#         return self.load_model(fold=best_fold)

#     def save_model_summary(self, model):

#         if self.model_neural:
#             path = Path(self.model_current_path, "model_summary.txt")
#             if path.exists():
#                 os.remove(path)

#             #Save the model summary (layers) to a txt file
#             with open(path, 'w') as f:
#                 model.summary(print_fn=lambda x: f.write(x + '\n'))
                

#     def save_summary(self, report, metrics=["precision", "recall", "f1-score"]):

#         #For all metrics
#         for metric in metrics:

#             #Create a path to the summary file
#             summary_path = Path(self.models_folder, self.type, f"summary_{metric}.csv")
            
#             #Get the columns (targets) and the values (scores per category)
#             columns = report[metric].index.tolist()
#             values = report[metric].values

#             #Expand the values : it's possible that a class does not exist in the predicted values
#             expected_columns = COLUMNS + ["accuracy", "macro avg", "weighted avg"]
#             expected_values = []   
#             for column in expected_columns:
#                 if str(column) in columns:
#                     expected_values.append(report[metric].loc[str(column)])
#                 else:
#                     expected_values.append(0.0)

#             # Read the existing dataframe or create it
#             df = pd.read_csv(summary_path, delimiter=",", index_col=0) if summary_path.exists() else pd.DataFrame(index=expected_columns)

#             #Create or replace the scores obtained by the model
#             df[self.get_name()] = expected_values
            
#             #Save it back to the summary file
#             df.to_csv(summary_path)

#     def decode_labels(self, labels, encoder=None):
#         return encoder.inverse_transform(label)

