import streamlit as st
import streamlit.components.v1 as components

title = "Rakuten Classification."
sidebar_name = "Rapport"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")
    html_file = open("assets/rapport.html", "r")
    components.html(html_file.read(), width=1000, height=564)