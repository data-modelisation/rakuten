import streamlit as st
import pandas as pd
import numpy as np


from  PIL import Image

from joblib import dump, load
import math as mt


def charger_models(uploaded_file,image_url,designation,description):
  
    if ((uploaded_file is None)& (image_url=="") & ((designation !="")|(description!="") )):
        
        dict_text = {"text": " this is a swimmingpool","lang text": "en","translated text": "ceci est une piscine","cleaned text": "c est une piscine","encoded text": "cec est une piscin"}
        st.header("Modèle de texte:")
        st.header("Exploration des textes")
        st.write("Le texte est: ",dict_text['text'])
        st.write("Le texte traduit est: ",dict_text["translated text"])
        st.write("Le texte nettoyé est: ",dict_text["cleaned text"])
        response = {
  "inputs": [
    "costume batman",
    " poupée"
  ],
  "encoded predictions": [
    4,
    7
  ],
  "encoded trues": [
    -1,
    -1
  ],
  "confidences": [
    0.3459031879901886,
    0.4225611686706543
  ],
  "decoded predictions": [
    1140,
    1280
  ],
  "decoded trues": [
    -1,
    -1
  ],
  "named predictions": [
    "Figurine",
    "Déguisement"
  ],
  "named trues": [
    "na",
    "na"
  ],
  "value probas": [
    [
      0.051511120051145554,
      0.10320615768432617,
      0.009604889899492264,
      0.0003952668921556324,
      0.3459031879901886,
      0.09885801374912262,
      0.06440729647874832,
      0.11488717049360275,
      0.04890555888414383,
      0.01972213387489319,
      0.0007867608219385147,
      0.00889462511986494,
      0.028025934472680092,
      0.0008417235221713781,
      0.0016209626337513328,
      0.0018339060479775071,
      0.0037359041161835194,
      0.0011142524890601635,
      0.00938958115875721,
      0.03448403999209404,
      0.024398544803261757,
      0.009667370468378067,
      0.0005438873777166009,
      0.000857608625665307,
      0.0019872216507792473,
      0.013849793933331966,
      0.0005671036778949201
    ],
    [
      0.013426045887172222,
      0.05412735044956207,
      0.009110216982662678,
      0.0002224212948931381,
      0.1544184535741806,
      0.010459376499056816,
      0.032720234245061874,
      0.4225611686706543,
      0.07204671204090118,
      0.07325644046068192,
      0.0010932825971394777,
      0.015064680017530918,
      0.06856560707092285,
      0.0033063837327063084,
      0.004017706960439682,
      0.0010578535730019212,
      0.01895844005048275,
      0.0014297212474048138,
      0.0056344871409237385,
      0.008703634142875671,
      0.005106584634631872,
      0.005411431659013033,
      0.0017646081978455186,
      0.0014542186399921775,
      0.007058658637106419,
      0.008804123848676682,
      0.00022021621407475322
    ]
  ],
  "named probas": [
    "Livre occasion",
    "Jeu vidéos",
    "Accessoire Console",
    "Consoles",
    "Figurine",
    "Carte Collection",
    "Jeu Plateau",
    "Déguisement",
    "Boite de jeu",
    "Jouet Tech",
    "Chaussette",
    "Gadget",
    "Bébé",
    "Salon",
    "Chambre",
    "Cuisine",
    "Chambre enfant",
    "Animaux",
    "Affiche",
    "Vintage",
    "Jeu oldschool",
    "Bureautique",
    "Décoration",
    "Aquatique",
    "Soin et Bricolage",
    "Livre neuf",
    "Jeu PC"
  ]
}       
        st.header("Résultats de la prédictions:")
        st.write("La catégorie prédicte est :",response["named predictions"][0])
        st.write("La probabilité prédicte est de "+str(mt.floor(response["confidences"][0]*100))+"%")
        st.progress(response["confidences"][0])
        
            


        st.header("Graphe des probabilités:")
       

        chart_data1 = pd.DataFrame(data=response["value probas"][0],index=response["named probas"],columns=['probabilités'])
        chart_data2 = pd.DataFrame(data=response["value probas"][1],index=response["named probas"],columns=['probabilités'])
        st.bar_chart(chart_data1)
        

        
    elif (((uploaded_file is not None)| (image_url!="")) & (designation=="") & (description=="")):
        
        st.header("Modèle d'image:")
        response = {
  "inputs": [
    "costume batman",
    " poupée"
  ],
  "encoded predictions": [
    4,
    7
  ],
  "encoded trues": [
    -1,
    -1
  ],
  "confidences": [
    0.3459031879901886,
    0.4225611686706543
  ],
  "decoded predictions": [
    1140,
    1280
  ],
  "decoded trues": [
    -1,
    -1
  ],
  "named predictions": [
    "Figurine",
    "Déguisement"
  ],
  "named trues": [
    "na",
    "na"
  ],
  "value probas": [
    [
      0.051511120051145554,
      0.10320615768432617,
      0.009604889899492264,
      0.0003952668921556324,
      0.3459031879901886,
      0.09885801374912262,
      0.06440729647874832,
      0.11488717049360275,
      0.04890555888414383,
      0.01972213387489319,
      0.0007867608219385147,
      0.00889462511986494,
      0.028025934472680092,
      0.0008417235221713781,
      0.0016209626337513328,
      0.0018339060479775071,
      0.0037359041161835194,
      0.0011142524890601635,
      0.00938958115875721,
      0.03448403999209404,
      0.024398544803261757,
      0.009667370468378067,
      0.0005438873777166009,
      0.000857608625665307,
      0.0019872216507792473,
      0.013849793933331966,
      0.0005671036778949201
    ],
    [
      0.013426045887172222,
      0.05412735044956207,
      0.009110216982662678,
      0.0002224212948931381,
      0.1544184535741806,
      0.010459376499056816,
      0.032720234245061874,
      0.4225611686706543,
      0.07204671204090118,
      0.07325644046068192,
      0.0010932825971394777,
      0.015064680017530918,
      0.06856560707092285,
      0.0033063837327063084,
      0.004017706960439682,
      0.0010578535730019212,
      0.01895844005048275,
      0.0014297212474048138,
      0.0056344871409237385,
      0.008703634142875671,
      0.005106584634631872,
      0.005411431659013033,
      0.0017646081978455186,
      0.0014542186399921775,
      0.007058658637106419,
      0.008804123848676682,
      0.00022021621407475322
    ]
  ],
  "named probas": [
    "Livre occasion",
    "Jeu vidéos",
    "Accessoire Console",
    "Consoles",
    "Figurine",
    "Carte Collection",
    "Jeu Plateau",
    "Déguisement",
    "Boite de jeu",
    "Jouet Tech",
    "Chaussette",
    "Gadget",
    "Bébé",
    "Salon",
    "Chambre",
    "Cuisine",
    "Chambre enfant",
    "Animaux",
    "Affiche",
    "Vintage",
    "Jeu oldschool",
    "Bureautique",
    "Décoration",
    "Aquatique",
    "Soin et Bricolage",
    "Livre neuf",
    "Jeu PC"
  ]
}       
        st.header("Résultats de la prédictions:")
        st.write("La catégorie prédicte est :",response["named predictions"][0])
        st.write("La probabilité prédicte est de "+str(mt.floor(response["confidences"][0]*100))+"%")
        st.progress(response["confidences"][0])
       

        st.header("Graphe des probabilités:")
       

        chart_data1 = pd.DataFrame(data=response["value probas"][0],index=response["named probas"],columns=['probabilités'])
        chart_data2 = pd.DataFrame(data=response["value probas"][1],index=response["named probas"],columns=['probabilités'])
        st.bar_chart(chart_data1)
        


    elif (((uploaded_file is not None)| (image_url!="")) & ((designation !="")|(description!="") )):
        
        dict_text = {"text": " this is a swimmingpool","lang text": "en","translated text": "ceci est une piscine","cleaned text": "c est une piscine","encoded texts": "cec est une piscin"}
        st.header("Modèle de fusion:")
        st.header("Exploration des textes")
        st.write("Le texte est: ",dict_text['text'])
        st.write("Le texte traduit est: ",dict_text["translated text"])
        st.write("Le texte nettoyé est: ",dict_text["cleaned text"])
        response = {
  "inputs": [
    "costume batman",
    " poupée"
  ],
  "encoded predictions": [
    4,
    7
  ],
  "encoded trues": [
    -1,
    -1
  ],
  "confidences": [
    0.3459031879901886,
    0.4225611686706543
  ],
  "decoded predictions": [
    1140,
    1280
  ],
  "decoded trues": [
    -1,
    -1
  ],
  "named predictions": [
    "Figurine",
    "Déguisement"
  ],
  "named trues": [
    "na",
    "na"
  ],
  "value probas": [
    [
      0.051511120051145554,
      0.10320615768432617,
      0.009604889899492264,
      0.0003952668921556324,
      0.3459031879901886,
      0.09885801374912262,
      0.06440729647874832,
      0.11488717049360275,
      0.04890555888414383,
      0.01972213387489319,
      0.0007867608219385147,
      0.00889462511986494,
      0.028025934472680092,
      0.0008417235221713781,
      0.0016209626337513328,
      0.0018339060479775071,
      0.0037359041161835194,
      0.0011142524890601635,
      0.00938958115875721,
      0.03448403999209404,
      0.024398544803261757,
      0.009667370468378067,
      0.0005438873777166009,
      0.000857608625665307,
      0.0019872216507792473,
      0.013849793933331966,
      0.0005671036778949201
    ],
    [
      0.013426045887172222,
      0.05412735044956207,
      0.009110216982662678,
      0.0002224212948931381,
      0.1544184535741806,
      0.010459376499056816,
      0.032720234245061874,
      0.4225611686706543,
      0.07204671204090118,
      0.07325644046068192,
      0.0010932825971394777,
      0.015064680017530918,
      0.06856560707092285,
      0.0033063837327063084,
      0.004017706960439682,
      0.0010578535730019212,
      0.01895844005048275,
      0.0014297212474048138,
      0.0056344871409237385,
      0.008703634142875671,
      0.005106584634631872,
      0.005411431659013033,
      0.0017646081978455186,
      0.0014542186399921775,
      0.007058658637106419,
      0.008804123848676682,
      0.00022021621407475322
    ]
  ],
  "named probas": [
    "Livre occasion",
    "Jeu vidéos",
    "Accessoire Console",
    "Consoles",
    "Figurine",
    "Carte Collection",
    "Jeu Plateau",
    "Déguisement",
    "Boite de jeu",
    "Jouet Tech",
    "Chaussette",
    "Gadget",
    "Bébé",
    "Salon",
    "Chambre",
    "Cuisine",
    "Chambre enfant",
    "Animaux",
    "Affiche",
    "Vintage",
    "Jeu oldschool",
    "Bureautique",
    "Décoration",
    "Aquatique",
    "Soin et Bricolage",
    "Livre neuf",
    "Jeu PC"
  ]
}       
        st.header("Résultats de la prédictions:")
        st.write("La catégorie prédicte est :",response["named predictions"][0])
        st.write("La probabilité prédicte est de "+str(mt.floor(response["confidences"][0]*100))+"%")
        st.progress(response["confidences"][0])
        

        st.header("Graphe des probabilités:")
       

        chart_data1 = pd.DataFrame(data=response["value probas"][0],index=response["named probas"],columns=['probabilités'])
        chart_data2 = pd.DataFrame(data=response["value probas"][1],index=response["named probas"],columns=['probabilités'])
        st.bar_chart(chart_data1)
        
    else:
        st.write("Rentrez des données svp.")


title = "Modèles"
sidebar_name = "Models"
Path_model = '../../src/models/'

def run():
    st.title(title)
    st.header("Rentrez un texte ou choisissez une image:")
    col1, col2 = st.columns(2)
    with col2:
        st.write("Téléchargez une image")
        uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image,width=300)
        image_url= st.text_input("Image URL")
        if image_url !="":
            st.write(image_url)
            st.image(image_url,width=300)
            
    with col1:
        st.write("Désignation")
        designation = st.text_input("")
        st.write("")
        st.write("")
        st.write("")
        description = st.text_area("Description")
  
    if st.button(label='validez votre choix de données'):
        charger_models(uploaded_file,image_url,designation,description)
    
    

   
        


         