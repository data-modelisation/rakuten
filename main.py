
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import pickle
from joblib import dump, load
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import src.tools.text as text_tools
import src.tools.image as image_tools
import src.tools.commons as commons
import src.tools.graphs as graphs
import src.models.models_utils as model_tools

PATH_RAW = "./data/raw/"
PATH_BASE = "./data/base/"
PATH_FEAT = "./data/featured/"
PATH_TRANS = "./data/translated/"
PATH_MODELS = "./src/models/"

MODEL_TEXT_NAME = "nn_simple"
MODEL_IMAGE_NAME = "cnn_simple"

BATCH_SIZE = 32
EPOCHS = 5
TARGET_SHAPE = (100, 100, 3)

# Creation d'un parser pour mettre des arguments en ligne de commande
parser = argparse.ArgumentParser(
                    prog = 'Rakuten Project',
                    description = 'Classification of products',
                    epilog = 'Enjoy!')

parser.add_argument('--lang', action='store_true', help='translate')
parser.add_argument('--train', action='store_true', help='train the model')

parser.add_argument('--text', action='store_true', help='work with text data')
parser.add_argument('--image', action='store_true', help='work with image data')

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

if __name__ == "__main__":
    
    start_time = time.time()
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

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test_tr = le.transform(y_test)
    
    #Si on travaille sur le texte
    if args.text:

        #Chemin du modèle  
        model_text_path = Path(PATH_MODELS, "text", MODEL_TEXT_NAME)
        model_text_path.mkdir(parents=True, exist_ok=True)

        #Entrainement d'un modele pour le texte avec l'option --train
        if args.train:
            start_time_text = time.time()

            #Transformations : vectorisation du texte en mots courant
            pipeline_preprocess = text_tools.build_pipeline_preprocessor(vectorize_type="tfidf")    
            X_train = pipeline_preprocess.fit_transform(X_train)
            stop_time_text_trans = time.time()
            print(f"model_text transformed in {stop_time_text_trans-start_time_text:03.2f}s")

            #Chargement du modele
            model_text, model_params = text_tools.build_pipeline_model(
                name=MODEL_TEXT_NAME,
                input_dim=X_train.shape[1])

            #Grilles pour GridSearchCV
            print(f"Paramètres : {model_params}")

            #Entrainement
            model_text.fit(X_train, y_train) 
            stop_time_text_train = time.time()
            print(f"model_text trained in {stop_time_text_train-stop_time_text_trans:03.2f}s")

            #Sauvegarde
            dump(model_text, Path(model_text_path, 'model.joblib'))
            dump(pipeline_preprocess, Path(model_text_path, 'preprocess.joblib'))
            stop_time_text_save = time.time()
            print(f"model_text saved in {stop_time_text_save-stop_time_text_train:03.2f}s")
            
        #Sinon chargement d'un modele pour le texte
        else:
            pass
            # model_text = load('src/models/model_text.joblib')
            # pipeline_preprocess = load('src/models/preprocess_text.joblib')
        
        #Prediction
        X_test = pipeline_preprocess.transform(X_test)
        y_text_preds = model_text.predict(X_test)
        
        if MODEL_TEXT_NAME.startswith("nn_"):
            y_text_preds = np.argmax(y_text_preds, axis=1)

        y_text_preds = le.inverse_transform(y_text_preds)

        crosstab = pd.crosstab(y_test, y_text_preds, rownames=["Real"], colnames=["Predicted"])
        print(classification_report(y_test, y_text_preds, zero_division=0))

        print(f"Text prediction finished in {time.time() - start_time:03.2f}s")
        heat = graphs.heatmap(crosstab)
        plt.savefig(Path(model_text_path, 'crosstab.jpg'))
        

    # Nettoyage de la figure
    plt.clf() 

    #Si on travaille sur les images
    if args.image:

        start_time_image = time.time()

        #Chemin du modèle  
        model_image_path = Path(PATH_MODELS, "image", MODEL_IMAGE_NAME)
        model_image_path.mkdir(parents=True, exist_ok=True)

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
        stop_time_image_read = time.time()
        print(f"images loaded in {stop_time_image_read-start_time_image:03.2f}s")

        #Entrainement d'un modele pour les images avec l'option --train
        if args.train:         

            #Transformations : gray + pca
            #pipeline_preprocess = image_tools.build_pipeline_preprocessor()    
            #Image_train = pipeline_preprocess.fit_transform(Image_train)
            stop_time_image_trans = time.time()
            print(f"model_image transformed in {stop_time_image_trans-stop_time_image_read:03.2f}s")

            model_image = image_tools.build_pipeline_model(
                name=MODEL_IMAGE_NAME,
                input_dim=TARGET_SHAPE)
            
            print(model_image.summary())
            training_history = model_image.fit_generator(
                generator=train_generator,
                epochs=EPOCHS,
                callbacks=[model_tools.get_model_checkpoint(model_image_path)]
            )
            #Entrainement
            #model_image = image_tools.build_pipeline_model()
            #model_image.fit(Image_train, y_train) 
            stop_time_image_train = time.time()
            print(f"model_image trained in {stop_time_image_train-stop_time_image_trans:03.2f}s")

            #Sauvegarde
            dump(model_image, Path(model_image_path, 'model.joblib'))
            stop_time_image_save = time.time()
            print(f"model_image saved in {stop_time_image_save-stop_time_image_train:03.2f}s")

        #Sinon chargement d'un modele pour le texte
        else:
            pass
            # model_image = load('src/models/model_image.joblib')
            # pipeline_preprocess = load('src/models/preprocess_image.joblib')
        
        #Prediction
        y_image_preds = model_image.predict(test_generator)
        y_image_preds = np.argmax(y_image_preds, axis=1)
        #y_test = y_test.transpose()

        y_image_preds = le.inverse_transform(y_image_preds)

        crosstab = pd.crosstab(y_test, y_image_preds, rownames=["Real"], colnames=["Predicted"])
        print(classification_report(y_test, y_image_preds, zero_division=0))

        print(f"Image prediction finished in {time.time() - start_time:03.2f}s")

        heat = graphs.heatmap(crosstab)
        plt.savefig(Path(model_image_path, 'crosstab.jpg'))