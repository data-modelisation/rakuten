import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from annotated_text import annotated_text
import re
import scrapper
import flag

title = "Let's predict things ! "
sidebar_name = "Prediction"

def clear_form():
    st.session_state["text_input"] = ""
    st.session_state["url_input"] = ""
    st.session_state["scrap_input"] = ""

def run():
    st.title(title)

    st.markdown("---")
    response={}
    C1, _, C2 = st.columns((10,2, 16))

    with C1:
        

        with st.form(key='columns_in_form'):
            
            text_input = st.text_area("Text", 
                key="text_input", 
                height=120,
                placeholder="Après dix-sept ans d'absence, Joe revient à Bush Falls, le patelin de son enfance. Couronné par le succès d'un livre qui ridiculisait ses voisins, il se heurte à l'hostilité d'une ville entière, bien décidée à lui faire payer ses écarts autobiographiques. Entre souvenirs et fantômes du passé, Joe va devoir affronter ses propres contradictions et peut-être enfin trouver sa place.'Mélanger ainsi humour et nostalgie est une prouesse rare, un vrai délice ! 'Charlotte Roux, Page des Libraires")
            text_input = re.sub('[^A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02af\u1d00-\u1d25\u1d62-\u1d65\u1d6b-\u1d77\u1d79-\u1d9a\u1e00-\u1eff\u2090-\u2094\u2184-\u2184\u2488-\u2490\u271d-\u271d\u2c60-\u2c7c\u2c7e-\u2c7f\ua722-\ua76f\ua771-\ua787\ua78b-\ua78c\ua7fb-\ua7ff\ufb00-\ufb06]', ' ', text_input)

            url_input = st.text_input("Image URL", key="url_input")

            scrap_input = st.text_input("Scrap URL", key="scrap_input", placeholder="Rueducommerce?")

            submit_button = st.form_submit_button(label='Predict 📈')
            clear = st.form_submit_button(label='Clean 🗑️', on_click=clear_form)
            
            if submit_button:
                if not scrap_input and not url_input and not text_input:
                    st.info('It would be easier with some inputs ... ', icon="ℹ️")
                else:
                    pass

                if scrap_input:
                    scrap_response = scrapper.scrap(scrap_input)
                    text_input = scrap_response.get("text_input", "")
                    text_input = re.sub(r'\W+', ' ', text_input)
                    url_input = scrap_response.get("url_input", "")

                    if not url_input:
                        st.warning(f"No image url found for {scrap_response.get('provider')}")
            
            with C2:                
                with st.spinner('Classifying, please wait....'):
                    try:
                        if text_input and not url_input:
                            print(f"asking FastAPI to predict this Text : {text_input}")
                            response = requests.get(f"http://127.0.0.1:8008/api/text/predict/text={text_input}").json()

                        elif not text_input and url_input:
                            print(f"asking FastAPI to predict this URL : {url_input}")
                            response = requests.get(f"http://127.0.0.1:8008/api/image/predict/url={url_input}").json()

                        elif text_input and url_input :
                            print(f"asking FastAPI to predict this Text {text_input} and this URL {url_input}")
                            response = requests.get(f"http://127.0.0.1:8008/api/fusion/predict/text={text_input}&url={url_input}").json()

                    except Exception as exce:
                        st.error("The backend doesn't respond ... Please wait or reload it ;)")
                        #st.error(exce)

            if response:
                if "activation_0" in response.keys():
                    with st.expander("Détails des images"):
                        image = np.array(response.get("activation_0"))
                        image = np.expand_dims(image, axis=2)
                        image = (image - image.min()) / (image.max() - image.min())
                        st.image(image, clamp=True)
                        

                if "annotated texts" in response.keys():
                    with st.expander("Détails du texte"):
                                
                        st.caption("Langue détectée")
                        lang = response.get("lang texts")[0].upper()
                        st.text(f"{lang}  {flag.flag(lang)}")
                        st.caption("Présence du texte dans la couche de vectorisation")
                        
                        annotated_tuple = [text if isinstance(text, str) else tuple(text) for text in response.get("annotated texts") ]
                        annotated_text(*annotated_tuple)
                        st.text_area("Texte traduit", response.get("translated texts")[0])
                        st.text_area("Texte nettoyé", response.get("cleaned texts")[0])

                with st.expander("Détails JSON"):
                    st.write(response)


    with C2:                
        if response:
                    
                    
                    st.progress(response["confidences"][0], text=f"{response['named predictions'][0]} | Probabilité {response['confidences'][0]*100:02.2f}%")

                    #Creation du graphique pour les probailités sur toutes les classes  
                    df = pd.DataFrame(response.get("value probas"), columns=response.get("named probas")).reset_index()
                    df = df.melt(id_vars=['index'], var_name="Catégories [-]", value_name="Probabilité [-]")
                    df["Domaine"] = response.get("macro named probas")
                    fig = px.bar(df, x='Catégories [-]', y='Probabilité [-]', color="Domaine", barmode='group')
                    for data in fig.data:
                        data["width"] = 0.55 #Change this value for bar widths
                    st.plotly_chart(fig, use_container_width=True)



    # tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    # st.sidebar.markdown("---")
    # st.sidebar.markdown(f"## {config.PROMOTION}")

    # st.sidebar.markdown("### Team members:")
    # for member in config.TEAM_MEMBERS:
    #     st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)
