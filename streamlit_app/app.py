from collections import OrderedDict
import requests
import streamlit as st
import pandas as pd
import numpy as np
# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config
import plotly.express as px

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import intro, second_tab, third_tab


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (second_tab.sidebar_name, second_tab),
        (third_tab.sidebar_name, third_tab),
    ]
)


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
  ],
    "texts": [
    "ceci est une piscine",
    " this is a swimmingpool"
  ],
"lang texts": [
    "fr",
    "en"
  ],
"translated texts": [
    "ceci est une piscine",
    "c'est une piscine"
  ],
"cleaned texts": [
    "ceci est une piscine",
    "c est une piscine"
  ],
"encoded texts": [
    "cec est une piscin",
    "c est une piscin"
]
}

def run():

    st.title("Rakuten Classification")

    with st.form(key='columns_in_form'):
        c1, c2 = st.columns(2)
        with c1:
            text_input = st.text_area("Text", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
        

        with c2:
            url_input = st.text_input("Image")

        submit_button = st.form_submit_button(label='Make a guess')
        if submit_button:
            
            #print(f"asking fastpi about this : {url}")
            #res = requests.get(f"http://127.0.0.1:8008/api/image/predict/url/{url}")
            #print(f"response from fastpi : {res}")

            #response = res.json()
            # with st.spinner('Classifying the image, please wait....'):
            #     st.write(response)

            # group_labels = ["Image",]

            # print(f"asking fastpi : {message}")
            # #res = requests.get(f"http://127.0.0.1:8008/api/text/predict/{message}")
            # #print(f"response from fastpi : {res}")

            # #response = res.json()

            cols = st.columns(1)
            for idx, col in enumerate(cols):
            prediction = response.get('named predictions')[idx]
            proba = round(response.get('confidences')[idx]*100,1)
            st.metric("Prédiction", prediction, delta=proba, delta_color="normal", help=None, label_visibility="visible")
            
            with st.expander("See explanation"):
                st.write(\"\"\"
                    The chart above shows some numbers I picked for you.
                    I rolled actual dice for these, so they're *guaranteed* to
                    be random.
                \"\"\")
                st.image("https://static.streamlit.io/examples/dice.jpg")

            group_labels = response.get("inputs")
            
            
            df = pd.DataFrame(response.get("value probas"), columns=response.get("named probas"))
            df["input"] = response.get("inputs")
            df = df.melt(id_vars=['input'])

            fig = px.bar(df, x='variable', y='value', color="input", barmode='group')
            fig.update_layout(legend=dict(x=0, y=1.25))
            st.plotly_chart(fig, use_container_width=True)

            with st.spinner('Classifying the text, please wait....'):
                st.write(response)

    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()




if __name__ == "__main__":
    run()
