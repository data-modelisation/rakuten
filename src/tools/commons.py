import pandas as pd
from pathlib import Path
import time

#Decorateur pour afficher le temps des fonctions
def timeit(fct):
    def timed_fct(*args, **kw):

        ts = time.time()
        result = fct(*args, **kw)
        te = time.time()

        print(f'func {fct.__name__} took: {te-ts:02.3f} sec')
        return result
    return timed_fct

# Fusion de colonnes de texte
def merge_columns(*args):
    merged_column = ""
    for column in args:
        merged_column += column + " "

    return merged_column

#Selection de samples
def select_samples(*args, samples=0):
    
    indexes = args[0].sample(n=samples).index.values

    return [arg.loc[indexes] for arg in args]

#Sauvegarde en pickle
def save_pkl(df, name="unamed.pkl", folder=""):

    if not Path(folder).exists():
        Path(folder).mkdir(parents=True, exist_ok=True)

    df.to_pickle(Path(folder, name))

#Lecture en pickle
def read_pkl(name="unamed.pkl", folder=""):
    return pd.read_pickle(Path(folder, name))

#Conversion des numéros des categories en description
def convert_to_readable_categories(serie:pd.Series):
    categories_num = [
        10, 40, 50, 60, 1140, 1160, 1180,
        1280, 1281, 1300, 1301, 1302, 1320, 1560,
        1920, 1940, 2060, 2220, 2280, 2403,
        2462, 2522, 2582, 2583, 2585, 2705, 2905
        ]

    categories_des = [
        "Livre", "Jeu Console", "Accessoire Console", "Tech", "Figurine", "Carte Collection", "Jeu Plateau",
        "Déguisement", "Boite de jeu", "Jouet Tech", "Chaussette", "Gadget", "Bébé", "Salon",
        "Chambre", "Cuisine", "Chambre enfant", "Animaux", "Affiche", "Vintage",
        "Jeu oldschool", "Bureautique", "Décoration", "Aquatique", "Soin et Bricolage", "Livre 2", "Jeu Console 2"
        ]

    categories_des_num = [f"{category_des}_{category_num}" for category_des, category_num in zip(categories_des, categories_num)]
    
    return serie.replace(
        to_replace = categories_num ,
        value = categories_des_num
    )