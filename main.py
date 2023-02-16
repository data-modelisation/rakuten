
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
import src.tools.text as text_tools
import src.tools.image as image_tools
import src.tools.commons as commons
import src.tools.graphs as graphs
import src.models.models_utils as model_tools
import src.models.models_fusion as model_fusion

import keras
import tensorflow as tf

PATH_PROJECT = "."
PATH_RAW = Path(PATH_PROJECT, "data/raw")
PATH_BASE = Path(PATH_PROJECT, "data/base")
PATH_FEAT = Path(PATH_PROJECT, "data/featured")
PATH_TRANS = Path(PATH_PROJECT, "data/translated")
PATH_MODELS = Path(PATH_PROJECT, "src/models")

MODEL_TEXT_NAME = "nn_simple"
MODEL_IMAGE_NAME = "cnn_simple"
MODEL_FUSION_NAME = "fusion_simple"

TRAIN_IMAGE = True
TRAIN_TEXT = True
TRAIN_FUSION = False

BATCH_SIZE = 32
EPOCHS_IMAGE = 10
EPOCHS_TEXT = 10
TARGET_SHAPE = (224, 224, 3)

LOG_DIR_TEXT = Path(PATH_PROJECT, "logs/text", MODEL_TEXT_NAME)
LOG_DIR_IMAGE = Path(PATH_PROJECT, "logs/image",MODEL_IMAGE_NAME)
LOG_DIR_FUSION = Path(PATH_PROJECT, "logs/fusion",MODEL_FUSION_NAME)

if __name__ == "__main__":
    
    # Creation d'un parser pour mettre des arguments en ligne de commande
    parser = argparse.ArgumentParser(
                        prog = 'Rakuten Project',
                        description = 'Classification of products',
                        epilog = 'Enjoy!')

    parser.add_argument('--lang', action='store_true', help='translate')
    parser.add_argument('--train', action='store_true', help='train the model')

    parser.add_argument('--text', action='store_true', help='work with text data')
    parser.add_argument('--image', action='store_true', help='work with image data')
    parser.add_argument('--fusion', action='store_true', help='work with text data')
    parser.add_argument('--train-text', action='store_true', help='work with image data')
    parser.add_argument('--train-image', action='store_true', help='work with text data')
    parser.add_argument('--train-fusion', action='store_true', help='work with image data')
    
    parser.add_argument('--samples', '--samples',
                        default=0,
                        help='how many samples?',
                        type=int
                        )
    parser.add_argument('--test-size', '--test-size',
                        default=.15,
                        help='how to split?',
                        type=float
                        )
    args = parser.parse_args()

    #Chargement des targets
    df_y = text_tools.read_csv(name="Y_train_CVw08PX.csv", folder=PATH_RAW)

    #Si on a pas l'option --lang, on charge directement le dataframe traduit et les targets avec noms
    if not args.lang:
        df_text = commons.read_pkl(name="df_text.pkl", folder=PATH_TRANS)
        
    else:
        #Chargement des observations textuelles
        df_text = text_tools.read_csv(name="X_train_update.csv", folder=PATH_RAW)
        
        #Ajout des noms des categories TODO
        #df_y["prdtypename"] = commons.convert_to_readable_categories(df_y.prdtypecode)

        #Transformations : creation des liens, comptage des mots et
        #longueurs des textes et suppression des colonnes inutiles
        pipeline_loader = text_tools.build_pipeline_load()    
        df_text = pipeline_loader.fit_transform(df_text)
        
        #Transformations : recherche de la langue et traduction
        pipeline_lang = text_tools.build_pipeline_lang(translate=False)    #TODO régler le pb avec l'API de trad
        df_text = pipeline_lang.fit_transform(df_text)
        #lang |text(traduit)

        commons.save_pkl(df_text, name="df_text.pkl", folder=PATH_TRANS)
        #commons.save_pkl(df_y, name="df_y.pkl", folder=PATH_TRANS) TODO

    print("datasets loaded")

    #Si on choisit de travailler sur quelques samples avec l'option --samples
    if args.samples > 0:
        df_text = df_text.sample(n=args.samples)
        df_y = df_y.loc[df_text.index]
        print("datasets sampled")
    
    #Séparation des données en entrainement et test
    X_train, X_test, y_train, y_test = train_test_split(
        df_text,
        df_y.prdtypecode,
        test_size=args.test_size,
        stratify=df_y.prdtypecode,
        )
    print("datasets splitted")  
    
    #Equilibrage du dataset
    #ro = RandomUnderSampler()
    #X_train, y_train = ro.fit_resample(X_train, y_train)

    #Passage en Numpy
    X_train, y_train = X_train.values, y_train.values
    X_test, y_test = X_test.values, y_test.values

    #Conversion des labels de la target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_tr = le.fit_transform(y_train)
    y_test_tr = le.transform(y_test)
    
    #Si on travaille sur le texte
    if args.text:

        #Chemin du modèle des texts 
        MODEL_TEXT_PATH = Path(PATH_MODELS, "text", MODEL_TEXT_NAME)
        MODEL_TEXT_PATH.mkdir(parents=True, exist_ok=True)

        #Transformations : vectorisation du texte en mots courant
        pipeline_preprocess_text = text_tools.build_pipeline_preprocessor(vectorize_type="tfidf")    
        
        if args.train_text:
            #Transformations : vectorisation du texte en mots courant
            X_train_tr = pipeline_preprocess_text.fit_transform(X_train)
            X_test_tr = pipeline_preprocess_text.transform(X_test)
            
            #Chargement du modele
            model_text, model_params, model_kwargs = text_tools.build_pipeline_model(
                    name=MODEL_TEXT_NAME,
                    epochs=EPOCHS_TEXT,
                    verbose=0,
                    validation_data=(X_test_tr, y_test_tr),
                    callbacks=[
                        model_tools.get_model_checkpoint(MODEL_TEXT_PATH), 
                        model_tools.get_dashboard(LOG_DIR_TEXT),
                        model_tools.get_tqdm()],
                    input_dim=X_train_tr.shape[1])

            #Lancement du Tensorboard
            #%tensorboard --logdir $LOG_DIR_TEXT

            # Entrainement
            model_text_hist = model_text.fit(X_train_tr, y_train_tr,
                **model_kwargs
                ) 
            
            # Sauvegarde
            joblib.dump(pipeline_preprocess_text, Path(MODEL_TEXT_PATH, 'pipeline_preprocess.joblib'))
            joblib.dump(model_text, Path(MODEL_TEXT_PATH, 'model.joblib'))
            
            if "history" in vars(model_text_hist):
                plt.plot(model_text_hist.history['accuracy'])
                plt.plot(model_text_hist.history['val_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train'], loc='upper left')

                plt.savefig(Path(MODEL_TEXT_PATH, 'history.jpg'))
                #plt.show()
                plt.clf()
        else:
            model_text = joblib.load(Path(MODEL_TEXT_PATH, 'model.joblib'))
            pipeline_preprocess_text = joblib.load(Path(MODEL_TEXT_PATH, 'pipeline_preprocess.joblib'))
        
        #Prediction
        X_test_tr = pipeline_preprocess_text.transform(X_test)
        y_text_preds = model_text.predict(X_test_tr)
        y_text_preds_class = np.argmax(y_text_preds, axis=1) if len(y_text_preds.shape)==2 else y_text_preds

        y_text_preds_class = le.inverse_transform(y_text_preds_class)

        crosstab = pd.crosstab(y_test, y_text_preds_class, rownames=["Real"], colnames=["Predicted"])
        print(classification_report_imbalanced(y_test, y_text_preds_class, zero_division=0))

        heat = graphs.heatmap(crosstab)
        plt.savefig(Path(MODEL_TEXT_PATH, 'crosstab.jpg'))
        

    # Nettoyage de la figure
    plt.clf() 

    #Si on travaille sur les images
    if args.image:
        
        #Chemin du modèle  
        MODEL_IMAGE_PATH = Path(PATH_MODELS, "image", MODEL_IMAGE_NAME)
        MODEL_IMAGE_PATH.mkdir(parents=True, exist_ok=True)

        #Génération des dataframes
        df_train = pd.DataFrame(
            data=np.concatenate([X_train[:,0].reshape(-1, 1), y_train.reshape(-1, 1)], axis=1),
            columns=["links", "label"])

        df_test = pd.DataFrame(
            data=np.concatenate([X_test[:,0].reshape(-1, 1), y_test_tr.reshape(-1, 1)], axis=1),
            columns=["links", "label"])
        
        df_train["label"] = df_train["label"].apply(lambda x: str(x))
        df_test["label"] = df_test["label"].apply(lambda x: str(x))
        
        #Creations des générateurs
        train_generator, test_generator = image_tools.flow_generators(
            df_train, df_test,
            TARGET_SHAPE, BATCH_SIZE
        )

        #Entrainement d'un modele pour les images avec l'option --train
        if args.train_image:    
            
            #Recuperation du modele
            model_image = image_tools.build_pipeline_model(
                name=MODEL_IMAGE_NAME,
                input_dim=TARGET_SHAPE,
            )
            
            #Lancement du Tensorboard
            #%tensorboard --logdir $LOG_IMAGE_TEXT

            #Entrainement
            model_image_hist = model_image.fit(
                train_generator,
                epochs=EPOCHS_IMAGE,
                verbose=0,
                validation_data=test_generator,
                callbacks=[
                        model_tools.get_model_checkpoint(MODEL_IMAGE_PATH), 
                        model_tools.get_dashboard(LOG_DIR_IMAGE),
                        model_tools.get_tqdm()
                        ]
            )

            #Sauvegarde
            #joblib.dump(model_image, Path(MODEL_IMAGE_PATH, 'model.joblib'))

            if "history" in vars(model_image):
                plt.plot(model_image_hist.history['accuracy'])
                plt.plot(model_image_hist.history['val_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                
                plt.savefig(Path(MODEL_IMAGE_PATH, 'history.jpg'))
                plt.clf()

        #Sinon chargement d'un modele pour le texte
        else:
            model_image = tf.keras.models.load_model(MODEL_IMAGE_PATH, compile=False)

        #Prediction
        y_image_preds = model_image.predict(test_generator)
        y_image_preds_class = np.argmax(y_image_preds, axis=1) if len(y_image_preds.shape)==2 else y_image_preds

        y_image_preds_class = le.inverse_transform(y_image_preds_class)

        crosstab = pd.crosstab(y_test, y_image_preds_class, rownames=["Real"], colnames=["Predicted"])
        print(classification_report_imbalanced(y_test, y_image_preds_class, zero_division=0))

        heat = graphs.heatmap(crosstab)
        plt.savefig(Path(MODEL_IMAGE_PATH, 'crosstab.jpg'))


    # model_fusion = fusion.model_multi_input(
    #     model_text, model_image
    # )
    # model_fusion.summary()

    # model_fusion_hist = model_fusion.fit(
    #     [X_train, train_generator],
    #     epochs=EPOCHS
    # )

    # y_fusion_preds = model_fusion.predict([X_test, test_generator])
    # y_fusion_preds_class = np.argmax(y_fusion_preds, axis=1)
    # y_fusion_preds_class = le.inverse_transform(y_fusion_preds_class)

    # crosstab = pd.crosstab(y_test, y_fusion_preds_class, rownames=["Real"], colnames=["Predicted"])
    # print(classification_report_imbalanced(y_test, y_image_preds, zero_division=0))
