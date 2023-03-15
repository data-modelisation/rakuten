import streamlit as st


title = "Rakuten Classification."
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        ## Rakuten

        * Site de e-commerce avec 1.3 milliards d'utilisateurs
        * Suggestions de recherche et recommandations pour l'utilisateur
        * Classification des produits nécessaire
        * Manuellement impossible

        ## Objectifs

        Prédire la catégorie d'un produit sur la base de son **titre**, sa **description** et de son **image**

        1 + 1 = 3 ... Un **modèle de texte**, un **modèle d'image** et un **modèle de fusion** 
        """
    )
