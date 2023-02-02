
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import pickle

import src.tools.text as text_tools
import src.tools.image as image_tools
import src.tools.commons as commons
import src.tools.graphs as graphs

PATH_RAW = "./data/raw/"
PATH_BASE = "./data/base/"
PATH_FEAT = "./data/featured/"

# Creation d'un parser pour mettre des arguments en ligne de commande
parser = argparse.ArgumentParser(
                    prog = 'Rakuten Project',
                    description = 'Classification of products',
                    epilog = 'Enjoy!')

parser.add_argument('--trans', action='store_true', help='transform')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--predict', action='store_true', help='make a prediction')
parser.add_argument('--show', action='store_true', help='show the predication in heatmap')

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
    df_text = text_tools.read_csv(name="X_train_update.csv", folder=PATH_RAW)
    df_y = text_tools.read_csv(name="Y_train_CVw08PX.csv", folder=PATH_RAW)
    print("dataset loaded")
    
    #Features and Target selection
    X_train, X_test, y_train, y_test = train_test_split(
        df_text,
        df_y.prdtypecode,
        test_size=args.test_size,
        stratify=df_y.prdtypecode,
        )
    print("dataset splitted")
    commons.save_pkl(y_train, name="y_train.pkl", folder=PATH_FEAT)
    commons.save_pkl(y_test, name="y_test.pkl", folder=PATH_FEAT)

    # Transformation des données
    if args.trans:
        preprocess = text_tools.build_preprocessor()
        X_train = preprocess.fit_transform(X_train)
        X_test = preprocess.transform(X_test)

        commons.save_pkl(X_train, name="X_train_transformed.pkl", folder=PATH_FEAT)
        commons.save_pkl(X_test, name="X_test_transformed.pkl", folder=PATH_FEAT)
        
        print("preprocessing finished")
    else:
        X_train = commons.read_pkl(name="X_train_transformed.pkl", folder=PATH_FEAT)
        X_test = commons.read_pkl(name="X_test_transformed.pkl", folder=PATH_FEAT)
        print("preprocessed datasets loaded")

    #Equilibrage du dataset
    rs = RandomOverSampler()
    X_train, y_train = rs.fit_resample(X_train, y_train)

    #Entrainement d'un modele
    if args.train:
        model = text_tools.build_pipeline()
        model.fit(X_train, y_train)
        pickle.dump(model, open("src/models/simple/knn/model_text_knn.pkl", 'wb'))
    
    else:    
        model = pickle.load(open("src/models/simple/knn/model_text_knn.pkl", 'rb'))

    #Prediction grâce au modele
    if args.predict:
        y_preds_test = model.predict(X_test)
        df_y_preds_test = pd.DataFrame(
            y_preds_test.reshape(-1,1),
            columns=["prdtypecode",],
            index=X_test.index)        

        print("prediction finished")
        commons.save_pkl(df_y_preds_test, name="y_preds_test.pkl", folder="src/models/simple/knn/")

    else:
        df_y_preds_test = commons.read_pkl(name="y_preds_test.pkl", folder="src/models/simple/knn/")
        print("predictions loaded")

    print(f"Finished in {time.time() - start_time}s")

    if args.show:
        y_preds_test = df_y_preds_test.prdtypecode
        crosstab = pd.crosstab(y_test, y_preds_test, rownames=["Real"], colnames=["Predicted"])
        print(classification_report(y_test, y_preds_test))
        heat = graphs.heatmap(crosstab)
        plt.show()