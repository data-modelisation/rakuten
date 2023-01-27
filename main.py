
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd

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

parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--predict', action='store_true', help='make a prediction')
parser.add_argument('--show', action='store_true', help='show the predication in heatmap')


parser.add_argument('--load-data',action='store_true', help='load the downloaded data')
parser.add_argument('--load-data-fe',action='store_true', help='load the feature engineering data')

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

    # Si on ne charge pas les données feature-engineering
    if not args.load_data_fe:

        # Si charge les données téléchargées
        if not args.load_data:

            #Lecture
            df_text = text_tools.read_csv(
                name="X_train_update.csv",
                folder=PATH_RAW)

            df_y = text_tools.read_csv(
                name="Y_train_CVw08PX.csv",
                folder=PATH_RAW)
            
            df_image = image_tools.read_image_files(
                df_text.productid,
                df_text.imageid,
                folder=PATH_RAW + "/images/image_train")

            #Conversion des numéros de catégorie en description
            df_y["prdtypename"] = commons.convert_to_readable_categories(df_y.prdtypecode)
            df_text["prdtypecode"] = df_y.prdtypecode
            df_text["prdtypename"] = df_y.prdtypename
            df_image["prdtypecode"] = df_y.prdtypecode
            df_image["prdtypename"] = df_y.prdtypename

            #Sauvegarde
            commons.save_pkl(df_text, name="df_text.pkl", folder=PATH_BASE)
            commons.save_pkl(df_y, name="df_y.pkl", folder=PATH_BASE)
        else:
            #Sinon on charge directement les données
            df_text = commons.read_pkl(name="df_text.pkl", folder=PATH_BASE)
            df_y = commons.read_pkl(name="df_y.pkl", folder=PATH_BASE)

        #Si on ne travaille que sur une selection
        if args.samples > 0:
            df_text, df_image, df_y = commons.select_samples(df_text, df_image, df_y, samples=args.samples)

        #On applique le feature-engineering sur les images et les textes
        df_text, df_commons = text_tools.apply_feature_engineering(
            df=df_text,
            stemm=True,
            translate=True)
        df_image = image_tools.apply_feature_engineering(df=df_image)

        #On sauvegarde
        for df, name in zip(
            (df_text, df_commons, df_y, df_image), 
            ("df_text", "df_commons", "df_y", "df_image")):
            commons.save_pkl(df, name=name+".pkl", folder=PATH_FEAT)
            print(f"sauvegarde de {name} terminée")

    else:
        #Sinon on charge les données déjà feature-engineering
        df_text = commons.read_pkl(name="df_text.pkl", folder=PATH_FEAT)    
        df_y = commons.read_pkl(name="df_y.pkl", folder=PATH_FEAT)

    #Features and Target selection
    features = df_text.drop(["lang", "designation", "description", "productid", "imageid", "prdtypecode", "prdtypename", "text", "text_clean", "text_stem"], axis=1)
    target = df_y.prdtypecode

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size
    )
    
    if args.train:
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)

    if args.predict:
        y_preds = model.predict(X_test)
        crosstab = pd.crosstab(y_test, y_preds, rownames=["Real"], colnames=["Predicted"])
        print(crosstab)

    if args.predict & args.show:
        heat = graphs.heatmap(crosstab)
        plt.show()
        
    print(f"Finished in {time.time() - start_time}s")
