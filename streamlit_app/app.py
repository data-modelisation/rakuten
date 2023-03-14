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

def run():

    st.title("Rakuten Classification")

    with st.form(key='columns_in_form'):
        c1, c2 = st.columns(2)
        with c1:
            text_input = st.text_area("Text", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
        
        with c2:
            url_input = st.text_input("Image URL")

        submit_button = st.form_submit_button(label='Make a guess')

        if submit_button:
            
            if text_input != "" and url_input == "":
              print(f"asking FastAPI to predict this Text : {text_input}")
              res = requests.get(f"http://127.0.0.1:8008/api/text/predict/{text_input}")
              print(f"response from fastpi : {res}")
            elif text_input == "" and url_input != "":
              print(f"asking FastAPI to predict this URL : {url_input}")
              res = requests.get(f"http://127.0.0.1:8008/api/image/predict/{url_input}")
              print(f"response from fastpi : {res}")
            elif text_input != "" and url_input != "":
              print(f"asking FastAPI to predict this Text {text_input} and this URL {url_input}")
              res = requests.get(f"http://127.0.0.1:8008/api/fusion/predict/{text_input}&url={url_input}")
              print(f"response from fastpi : {res}")

            response = res.json()
            with st.spinner('Classifying the image, please wait....'):
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
